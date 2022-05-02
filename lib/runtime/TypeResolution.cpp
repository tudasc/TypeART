// TypeART library
//
// Copyright (c) 2017-2022 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#include "TypeResolution.h"

#include "AllocationTracking.h"
#include "Runtime.h"
#include "RuntimeData.h"
#include "RuntimeInterface.h"
#include "support/Logger.h"
#include "support/System.h"

#include "llvm/ADT/Optional.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace typeart {

namespace detail {

template <typename P, typename T>
inline const void* add_byte_offset(const P* addr, T offset) {
  assert(addr != nullptr);
  return static_cast<const P*>(static_cast<const char*>(addr) + offset);
}

template <typename T>
inline std::ptrdiff_t substract_pointer(const T* addr_a, const T* addr_b) {
  assert(addr_a != nullptr);
  assert(addr_b != nullptr);
  return static_cast<const char*>(addr_a) - static_cast<const char*>(addr_b);
}

inline size_t get_member_index_at(const typeart_struct_layout& structInfo, size_t offset) {
  const size_t num_members = structInfo.num_members;

  if (num_members == 0) {
    return 0;
  }

  const auto* struct_offsets = structInfo.offsets;

  if (offset > struct_offsets[num_members - 1]) {
    return num_members - 1;
  }

  size_t member_index{0};
  while (member_index < (num_members - 1) && offset >= struct_offsets[member_index + 1]) {
    ++member_index;
  }
  return member_index;
}

}  // namespace detail

TypeResolution::TypeResolution(const TypeDB& type_db, Recorder& recorder) : typeDB{type_db}, recorder{recorder} {
}

TypeResolution::TypeArtStatus TypeResolution::getSubTypeInfo(const void* baseAddr, size_t offset,
                                                             const typeart_struct_layout& containerInfo, int* subType,
                                                             const void** subTypeBaseAddr, size_t* subTypeOffset,
                                                             size_t* subTypeCount) const {
  if (containerInfo.type_id < 0) {
    return TYPEART_ERROR;
  }

  if (offset >= containerInfo.extent) {
    return TYPEART_BAD_OFFSET;
  }

  // Get index of the struct member at the address
  const size_t memberIndex = detail::get_member_index_at(containerInfo, offset);
  const int memberType     = containerInfo.member_types[memberIndex];
  const size_t baseOffset  = containerInfo.offsets[memberIndex];

  const size_t internalOffset   = offset - baseOffset;
  const size_t typeSize         = typeDB.getTypeSize(memberType);
  const size_t offsetInTypeSize = internalOffset / typeSize;
  const size_t newOffset        = internalOffset % typeSize;

  // If newOffset != 0, the subtype cannot be atomic, i.e. must be a struct
  if (newOffset != 0) {
    if (typeDB.isReservedType(memberType)) {
      return TYPEART_BAD_ALIGNMENT;
    }
  }

  // Ensure that the array index is in bounds
  if (offsetInTypeSize >= containerInfo.count[memberIndex]) {
    // Address points to padding
    return TYPEART_BAD_ALIGNMENT;
  }

  *subType         = memberType;
  *subTypeBaseAddr = detail::add_byte_offset(baseAddr, baseOffset);
  *subTypeOffset   = newOffset;
  *subTypeCount    = containerInfo.count[memberIndex] - offsetInTypeSize;

  return TYPEART_OK;
}

TypeResolution::TypeArtStatus TypeResolution::getTypeInfoInternal(const void* baseAddr, size_t offset,
                                                                  const StructTypeInfo& containerInfo, int* type,
                                                                  size_t* count) const {
  assert(offset < containerInfo.extent && "Something went wrong with the base address computation");

  int subType{-1};
  const void* subTypeBaseAddr;
  size_t subTypeOffset{0};
  size_t subTypeCount{0};

  const StructTypeInfo* structInfo = &containerInfo;

  const auto to_struct_layout_info = [](const auto& struct_type_info) {
    typeart_struct_layout struct_layout;
    struct_layout.type_id      = struct_type_info.type_id;
    struct_layout.name         = struct_type_info.name.c_str();
    struct_layout.num_members  = struct_type_info.num_members;
    struct_layout.extent       = struct_type_info.extent;
    struct_layout.offsets      = &struct_type_info.offsets[0];
    struct_layout.member_types = &struct_type_info.member_types[0];
    struct_layout.count        = &struct_type_info.array_sizes[0];
    return struct_layout;
  };

  // Resolve type recursively, until the address matches exactly
  bool resolve{true};
  while (resolve) {
    const auto status_subtype_info = getSubTypeInfo(baseAddr, offset, to_struct_layout_info(*structInfo), &subType,
                                                    &subTypeBaseAddr, &subTypeOffset, &subTypeCount);

    if (status_subtype_info != TYPEART_OK) {
      return status_subtype_info;
    }

    baseAddr = subTypeBaseAddr;
    offset   = subTypeOffset;

    // Continue as long as there is a byte offset
    resolve = offset != 0;

    // Get layout of the nested struct
    if (resolve) {
      const auto status_struct_info = getStructInfo(subType, &structInfo);
      if (status_struct_info != TYPEART_OK) {
        return status_struct_info;
      }
    }
  }
  *type  = subType;
  *count = subTypeCount;
  return TYPEART_OK;
}

TypeResolution::TypeArtStatus TypeResolution::getTypeInfo(const void* addr, const void* basePtr,
                                                          const PointerInfo& ptrInfo, int* type, size_t* count) const {
  const int containing_type = ptrInfo.typeId;
  size_t containing_type_count{0};
  size_t internal_byte_offset{0};

  // First, retrieve the containing type
  TypeArtStatus status = getContainingTypeInfo(addr, basePtr, ptrInfo, &containing_type_count, &internal_byte_offset);
  if (status != TYPEART_OK) {
    if (status == TYPEART_UNKNOWN_ADDRESS) {
      recorder.incAddrMissing(addr);
    }
    return status;
  }

  // Check for exact address match
  if (internal_byte_offset == 0) {
    *type  = containing_type;
    *count = containing_type_count;
    return TYPEART_OK;
  }

  if (typeDB.isBuiltinType(containing_type)) {
    // Address points to the middle of a builtin type
    return TYPEART_BAD_ALIGNMENT;
  }

  // Resolve struct recursively
  const auto* structInfo = typeDB.getStructInfo(containing_type);
  if (structInfo != nullptr) {
    const void* containingTypeAddr = detail::add_byte_offset(addr, -std::ptrdiff_t(internal_byte_offset));
    return getTypeInfoInternal(containingTypeAddr, internal_byte_offset, *structInfo, type, count);
  }

  return TYPEART_INVALID_ID;
}

TypeResolution::TypeArtStatus TypeResolution::getContainingTypeInfo(const void* addr, const void* basePtr,
                                                                    const PointerInfo& ptrInfo, size_t* count,
                                                                    size_t* offset) const {
  const auto& basePtrInfo = ptrInfo;
  size_t typeSize         = typeDB.getTypeSize(basePtrInfo.typeId);

  // Check for exact match -> no further checks and offsets calculations needed
  if (basePtr == addr) {
    *count  = ptrInfo.count;
    *offset = 0;
    return TYPEART_OK;
  }

  // The address points inside a known array
  const void* blockEnd = detail::add_byte_offset(basePtr, basePtrInfo.count * typeSize);

  // Ensure that the given address is in bounds and points to the start of an element
  if (addr >= blockEnd) {
    const std::ptrdiff_t byte_offset_to_base = detail::substract_pointer(addr, basePtr);

    const auto elements  = byte_offset_to_base / typeSize;
    const auto oob_index = elements - basePtrInfo.count + 1;
    LOG_WARNING("Out of bounds for the lookup: (" << debug::toString(addr, basePtrInfo)
                                                  << ") #Elements too far: " << oob_index);
    return TYPEART_UNKNOWN_ADDRESS;
  }

  assert(addr >= basePtr && "Error in base address computation");

  const std::ptrdiff_t addrDif = detail::substract_pointer(addr, basePtr);

  // Offset of the pointer w.r.t. the start of the containing type
  const size_t internalOffset = addrDif % typeSize;

  // Array index
  const size_t typeOffset = addrDif / typeSize;
  const size_t typeCount  = basePtrInfo.count - typeOffset;

  // Retrieve and return type information
  *count  = typeCount;
  *offset = internalOffset;
  return TYPEART_OK;
}

TypeResolution::TypeArtStatus TypeResolution::getStructInfo(int type_id, const StructTypeInfo** structInfo) const {
  // Requested ID must correspond to a struct
  if (!typeDB.isStructType(type_id)) {
    return TYPEART_WRONG_KIND;
  }

  const auto* result = typeDB.getStructInfo(type_id);

  if (result != nullptr) {
    *structInfo = result;
    return TYPEART_OK;
  }
  return TYPEART_INVALID_ID;
}

const TypeDB& TypeResolution::db() const {
  return typeDB;
}

namespace detail {
inline typeart_status query_type(const void* addr, int* type, size_t* count) {
  auto alloc = typeart::RuntimeSystem::get().allocTracker.findBaseAlloc(addr);
  typeart::RuntimeSystem::get().recorder.incUsedInRequest(addr);
  if (alloc) {
    return typeart::RuntimeSystem::get().typeResolution.getTypeInfo(addr, alloc->first, alloc->second, type, count);
  }
  return TYPEART_UNKNOWN_ADDRESS;
}

inline typeart_status query_struct_layout(int type_id, typeart_struct_layout* struct_layout) {
  const typeart::StructTypeInfo* struct_info;
  typeart_status status = typeart::RuntimeSystem::get().typeResolution.getStructInfo(type_id, &struct_info);
  if (status == TYPEART_OK) {
    struct_layout->type_id      = struct_info->type_id;
    struct_layout->name         = struct_info->name.c_str();
    struct_layout->num_members  = struct_info->num_members;
    struct_layout->extent       = struct_info->extent;
    struct_layout->offsets      = &struct_info->offsets[0];
    struct_layout->member_types = &struct_info->member_types[0];
    struct_layout->count        = &struct_info->array_sizes[0];
  } else {
    struct_layout->type_id      = std::numeric_limits<decltype(typeart_struct_layout::type_id)>::min();
    struct_layout->name         = "";
    struct_layout->num_members  = 0;
    struct_layout->extent       = 0;
    struct_layout->offsets      = nullptr;
    struct_layout->member_types = nullptr;
    struct_layout->count        = nullptr;
  }
  return status;
}

char* string2char(std::string_view src) {
  const void* ret_addr       = __builtin_return_address(0);
  const size_t source_length = src.size() + 1;  // +1 for '\0'
  char* string_copy          = (char*)malloc(sizeof(char) * source_length);

  if (string_copy == nullptr) {
    return nullptr;
  }

  typeart::RuntimeSystem::get().allocTracker.onAlloc(string_copy, TYPEART_INT8, source_length, ret_addr);

  memcpy(string_copy, src.data(), source_length);

  return string_copy;
}

}  // namespace detail
}  // namespace typeart

/**
 * Runtime interface implementation
 *
 */

typeart_status typeart_get_type(const void* addr, int* type_id, size_t* count) {
  typeart::RTGuard guard;
  return typeart::detail::query_type(addr, type_id, count);
}

typeart_status typeart_get_type_length(const void* addr, size_t* count) {
  typeart::RTGuard guard;
  int type{0};
  return typeart::detail::query_type(addr, &type, count);
}

typeart_status typeart_get_type_id(const void* addr, int* type_id) {
  typeart::RTGuard guard;
  size_t count{0};
  return typeart::detail::query_type(addr, type_id, &count);
}

typeart_status typeart_get_containing_type(const void* addr, int* type_id, size_t* count, const void** base_address,
                                           size_t* byte_offset) {
  typeart::RTGuard guard;
  auto alloc = typeart::RuntimeSystem::get().allocTracker.findBaseAlloc(addr);
  if (alloc) {
    //    auto& allocVal = alloc.getValue();
    *type_id      = alloc->second.typeId;
    *base_address = alloc->first;
    return typeart::RuntimeSystem::get().typeResolution.getContainingTypeInfo(addr, alloc->first, alloc->second, count,
                                                                              byte_offset);
  }
  return TYPEART_UNKNOWN_ADDRESS;
}

typeart_status typeart_get_subtype(const void* base_addr, size_t offset, const typeart_struct_layout* container_layout,
                                   int* subtype_id, const void** subtype_base_addr, size_t* subtype_byte_offset,
                                   size_t* subtype_count) {
  typeart::RTGuard guard;
  auto status = typeart::RuntimeSystem::get().typeResolution.getSubTypeInfo(
      base_addr, offset, *container_layout, subtype_id, subtype_base_addr, subtype_byte_offset, subtype_count);
  return status;
}

typeart_status typeart_resolve_type_addr(const void* addr, typeart_struct_layout* struct_layout) {
  typeart::RTGuard guard;
  int type_id{0};
  size_t size{0};
  auto status = typeart::detail::query_type(addr, &type_id, &size);
  if (status != TYPEART_OK) {
    return status;
  }
  return typeart::detail::query_struct_layout(type_id, struct_layout);
}

typeart_status typeart_resolve_type_id(int type_id, typeart_struct_layout* struct_layout) {
  typeart::RTGuard guard;
  return typeart::detail::query_struct_layout(type_id, struct_layout);
}

typeart_status typeart_get_return_address(const void* addr, const void** return_addr) {
  typeart::RTGuard guard;
  auto alloc = typeart::RuntimeSystem::get().allocTracker.findBaseAlloc(addr);

  if (alloc) {
    *return_addr = alloc.getValue().second.debug;
    return TYPEART_OK;
  }
  *return_addr = nullptr;
  return TYPEART_UNKNOWN_ADDRESS;
}

typeart_status_t typeart_get_source_location(const void* addr, char** file, char** function, char** line) {
  using namespace typeart::detail;
  typeart::RTGuard guard;

  auto source_loc = typeart::SourceLocation::create(addr);

  if (source_loc) {
    *file     = string2char(source_loc->file);
    *function = string2char(source_loc->function);
    *line     = string2char(source_loc->line);

    if (*file == nullptr || *function == nullptr || *line == nullptr) {
      return TYPEART_ERROR;
    }

    return TYPEART_OK;
  }

  return TYPEART_UNKNOWN_ADDRESS;
}

const char* typeart_get_type_name(int type_id) {
  typeart::RTGuard guard;
  return typeart::RuntimeSystem::get().typeResolution.db().getTypeName(type_id).c_str();
}

bool typeart_is_vector_type(int type_id) {
  typeart::RTGuard guard;
  return typeart::RuntimeSystem::get().typeResolution.db().isVectorType(type_id);
}

bool typeart_is_valid_type(int type_id) {
  typeart::RTGuard guard;
  return typeart::RuntimeSystem::get().typeResolution.db().isValid(type_id);
}

bool typeart_is_reserved_type(int type_id) {
  typeart::RTGuard guard;
  return typeart::RuntimeSystem::get().typeResolution.db().isReservedType(type_id);
}

bool typeart_is_builtin_type(int type_id) {
  typeart::RTGuard guard;
  return typeart::RuntimeSystem::get().typeResolution.db().isBuiltinType(type_id);
}

bool typeart_is_struct_type(int type_id) {
  typeart::RTGuard guard;
  return typeart::RuntimeSystem::get().typeResolution.db().isStructType(type_id);
}

bool typeart_is_userdefined_type(int type_id) {
  typeart::RTGuard guard;
  return typeart::RuntimeSystem::get().typeResolution.db().isUserDefinedType(type_id);
}

size_t typeart_get_type_size(int type_id) {
  typeart::RTGuard guard;
  return typeart::RuntimeSystem::get().typeResolution.db().getTypeSize(type_id);
}

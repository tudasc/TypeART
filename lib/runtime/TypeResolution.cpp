//
// Created by sebastian on 07.01.21.
//

#include "TypeResolution.h"

#include "AccessCountPrinter.h"
#include "Runtime.h"

namespace typeart {

template <typename T>
inline const void* addByteOffset(const void* addr, T offset) {
  return static_cast<const void*>(static_cast<const uint8_t*>(addr) + offset);
}

TypeResolution::TypeResolution(const TypeDB& db, Recorder& recorder) : typeDB{db}, recorder{recorder} {
}

size_t TypeResolution::getMemberIndex(typeart_struct_layout structInfo, size_t offset) const {
  size_t n = structInfo.len;
  if (offset > structInfo.offsets[n - 1]) {
    return n - 1;
  }

  size_t i = 0;
  while (i < n - 1 && offset >= structInfo.offsets[i + 1]) {
    ++i;
  }
  return i;
}

TypeResolution::TypeArtStatus TypeResolution::getSubTypeInfo(const void* baseAddr, size_t offset,
                                                             typeart_struct_layout containerInfo, int* subType,
                                                             const void** subTypeBaseAddr, size_t* subTypeOffset,
                                                             size_t* subTypeCount) const {
  if (offset >= containerInfo.extent) {
    return TA_BAD_OFFSET;
  }

  // Get index of the struct member at the address
  size_t memberIndex = getMemberIndex(containerInfo, offset);

  int memberType = containerInfo.member_types[memberIndex];

  size_t baseOffset = containerInfo.offsets[memberIndex];
  assert(offset >= baseOffset && "Invalid offset values");

  size_t internalOffset   = offset - baseOffset;
  size_t typeSize         = typeDB.getTypeSize(memberType);
  size_t offsetInTypeSize = internalOffset / typeSize;
  size_t newOffset        = internalOffset % typeSize;

  // If newOffset != 0, the subtype cannot be atomic, i.e. must be a struct
  if (newOffset != 0) {
    if (typeDB.isReservedType(memberType)) {
      return TA_BAD_ALIGNMENT;
    }
  }

  // Ensure that the array index is in bounds
  if (offsetInTypeSize >= containerInfo.count[memberIndex]) {
    // Address points to padding
    return TA_BAD_ALIGNMENT;
  }

  *subType         = memberType;
  *subTypeBaseAddr = addByteOffset(baseAddr, baseOffset);
  *subTypeOffset   = newOffset;
  *subTypeCount    = containerInfo.count[memberIndex] - offsetInTypeSize;

  return TA_OK;
}

TypeResolution::TypeArtStatus TypeResolution::getSubTypeInfo(const void* baseAddr, size_t offset,
                                                             const StructTypeInfo& containerInfo, int* subType,
                                                             const void** subTypeBaseAddr, size_t* subTypeOffset,
                                                             size_t* subTypeCount) const {
  typeart_struct_layout structLayout;
  structLayout.id           = containerInfo.id;
  structLayout.name         = containerInfo.name.c_str();
  structLayout.len          = containerInfo.numMembers;
  structLayout.extent       = containerInfo.extent;
  structLayout.offsets      = &containerInfo.offsets[0];
  structLayout.member_types = &containerInfo.memberTypes[0];
  structLayout.count        = &containerInfo.arraySizes[0];
  return getSubTypeInfo(baseAddr, offset, structLayout, subType, subTypeBaseAddr, subTypeOffset, subTypeCount);
}

TypeResolution::TypeArtStatus TypeResolution::getTypeInfoInternal(const void* baseAddr, size_t offset,
                                                                  const StructTypeInfo& containerInfo, int* type,
                                                                  size_t* count) const {
  assert(offset < containerInfo.extent && "Something went wrong with the base address computation");

  TypeArtStatus status;
  int subType;
  const void* subTypeBaseAddr;
  size_t subTypeOffset;
  size_t subTypeCount;
  const StructTypeInfo* structInfo = &containerInfo;

  bool resolve = true;

  // Resolve type recursively, until the address matches exactly
  while (resolve) {
    status = getSubTypeInfo(baseAddr, offset, *structInfo, &subType, &subTypeBaseAddr, &subTypeOffset, &subTypeCount);

    if (status != TA_OK) {
      return status;
    }

    baseAddr = subTypeBaseAddr;
    offset   = subTypeOffset;

    // Continue as long as there is a byte offset
    resolve = offset != 0;

    // Get layout of the nested struct
    if (resolve) {
      status = getStructInfo(subType, &structInfo);
      if (status != TA_OK) {
        return status;
      }
    }
  }
  *type  = subType;
  *count = subTypeCount;
  return TA_OK;
}

TypeResolution::TypeArtStatus TypeResolution::getTypeInfo(const void* addr, const void* basePtr,
                                                          const PointerInfo& ptrInfo, int* type, size_t* count) const {
  int containingType = ptrInfo.typeId;
  size_t containingTypeCount;
  size_t internalOffset;

  // First, retrieve the containing type
  TypeArtStatus status = getContainingTypeInfo(addr, basePtr, ptrInfo, &containingTypeCount, &internalOffset);
  if (status != TA_OK) {
    if (status == TA_UNKNOWN_ADDRESS) {
      recorder.incAddrMissing(addr);
    }
    return status;
  }

  // Check for exact address match
  if (internalOffset == 0) {
    *type  = containingType;
    *count = containingTypeCount;
    return TA_OK;
  }

  if (typeDB.isBuiltinType(containingType)) {
    // Address points to the middle of a builtin type
    return TA_BAD_ALIGNMENT;
  }

  // Resolve struct recursively
  const auto* structInfo = typeDB.getStructInfo(containingType);
  if (structInfo != nullptr) {
    const void* containingTypeAddr = addByteOffset(addr, -internalOffset);
    return getTypeInfoInternal(containingTypeAddr, internalOffset, *structInfo, type, count);
  }
  return TA_INVALID_ID;
}

TypeResolution::TypeArtStatus TypeResolution::getContainingTypeInfo(const void* addr, const void* basePtr,
                                                                    const PointerInfo& ptrInfo, size_t* count,
                                                                    size_t* offset) const {
  const auto& basePtrInfo = ptrInfo;
  size_t typeSize         = getTypeSize(basePtrInfo.typeId);

  // Check for exact match -> no further checks and offsets calculations needed
  if (basePtr == addr) {
    *count  = ptrInfo.count;
    *offset = 0;
    return TA_OK;
  }

  // The address points inside a known array
  const void* blockEnd = addByteOffset(basePtr, basePtrInfo.count * typeSize);

  // Ensure that the given address is in bounds and points to the start of an element
  if (addr >= blockEnd) {
    const std::ptrdiff_t offset = static_cast<const uint8_t*>(addr) - static_cast<const uint8_t*>(basePtr);
    const auto oob_index        = (offset / typeSize) - basePtrInfo.count + 1;
    LOG_ERROR("Out of bounds for the lookup: (" << debug::toString(addr, basePtrInfo)
                                                << ") #Elements too far: " << oob_index);
    return TA_UNKNOWN_ADDRESS;
  }

  assert(addr >= basePtr && "Error in base address computation");
  size_t addrDif = reinterpret_cast<size_t>(addr) - reinterpret_cast<size_t>(basePtr);

  // Offset of the pointer w.r.t. the start of the containing type
  size_t internalOffset = addrDif % typeSize;

  // Array index
  size_t typeOffset = addrDif / typeSize;
  size_t typeCount  = basePtrInfo.count - typeOffset;

  // Retrieve and return type information
  *count  = typeCount;
  *offset = internalOffset;
  return TA_OK;
}

TypeResolution::TypeArtStatus TypeResolution::getBuiltinInfo(const void* addr, const PointerInfo& ptrInfo,
                                                             BuiltinType* type) const {
  if (typeDB.isReservedType(ptrInfo.typeId)) {
    *type = static_cast<BuiltinType>(ptrInfo.typeId);
    return TA_OK;
  }
  return TA_WRONG_KIND;
}

TypeResolution::TypeArtStatus TypeResolution::getStructInfo(int id, const StructTypeInfo** structInfo) const {
  // Requested ID must correspond to a struct
  if (!typeDB.isStructType(id)) {
    return TA_WRONG_KIND;
  }

  const auto* result = typeDB.getStructInfo(id);

  if (result != nullptr) {
    *structInfo = result;
    return TA_OK;
  }
  return TA_INVALID_ID;
}

const std::string& TypeResolution::getTypeName(int id) const {
  return typeDB.getTypeName(id);
}

size_t TypeResolution::getTypeSize(int id) const {
  return typeDB.getTypeSize(id);
}

bool TypeResolution::isValidType(int id) const {
  return typeDB.isValid(id);
}

}  // namespace typeart

/**
 * Runtime interface implementation
 *
 */

typeart_status typeart_get_builtin_type(const void* addr, typeart::BuiltinType* type) {
  typeart::RTGuard guard;
  auto alloc = typeart::RuntimeSystem::get().allocTracker.findBaseAlloc(addr);
  if (alloc) {
    return typeart::RuntimeSystem::get().typeResolution.getBuiltinInfo(addr, alloc->second, type);
  }
  return TA_UNKNOWN_ADDRESS;
}

typeart_status typeart_get_type(const void* addr, int* type, size_t* count) {
  typeart::RTGuard guard;
  auto alloc = typeart::RuntimeSystem::get().allocTracker.findBaseAlloc(addr);
  typeart::RuntimeSystem::get().recorder.incUsedInRequest(addr);
  if (alloc) {
    return typeart::RuntimeSystem::get().typeResolution.getTypeInfo(addr, alloc->first, alloc->second, type, count);
  }
  return TA_UNKNOWN_ADDRESS;
}

typeart_status typeart_get_containing_type(const void* addr, int* type, size_t* count, const void** base_address,
                                           size_t* offset) {
  typeart::RTGuard guard;
  auto alloc = typeart::RuntimeSystem::get().allocTracker.findBaseAlloc(addr);
  if (alloc) {
    auto& allocVal = alloc.getValue();
    *type          = alloc->second.typeId;
    *base_address  = alloc->first;
    return typeart::RuntimeSystem::get().typeResolution.getContainingTypeInfo(addr, alloc->first, alloc->second, count,
                                                                              offset);
  }
  return TA_UNKNOWN_ADDRESS;
}

typeart_status typeart_get_subtype(const void* base_addr, size_t offset, typeart_struct_layout container_layout,
                                   int* subtype, const void** subtype_base_addr, size_t* subtype_offset,
                                   size_t* subtype_count) {
  typeart::RTGuard guard;
  return typeart::RuntimeSystem::get().typeResolution.getSubTypeInfo(base_addr, offset, container_layout, subtype,
                                                                     subtype_base_addr, subtype_offset, subtype_count);
}

typeart_status typeart_resolve_type(int id, typeart_struct_layout* struct_layout) {
  typeart::RTGuard guard;
  const typeart::StructTypeInfo* structInfo;
  typeart_status status = typeart::RuntimeSystem::get().typeResolution.getStructInfo(id, &structInfo);
  if (status == TA_OK) {
    struct_layout->id           = structInfo->id;
    struct_layout->name         = structInfo->name.c_str();
    struct_layout->len          = structInfo->numMembers;
    struct_layout->extent       = structInfo->extent;
    struct_layout->offsets      = &structInfo->offsets[0];
    struct_layout->member_types = &structInfo->memberTypes[0];
    struct_layout->count        = &structInfo->arraySizes[0];
  }
  return status;
}

const char* typeart_get_type_name(int id) {
  typeart::RTGuard guard;
  return typeart::RuntimeSystem::get().typeResolution.getTypeName(id).c_str();
}

void typeart_get_return_address(const void* addr, const void** retAddr) {
  typeart::RTGuard guard;
  auto alloc = typeart::RuntimeSystem::get().allocTracker.findBaseAlloc(addr);

  if (alloc) {
    *retAddr = alloc.getValue().second.debug;
  } else {
    *retAddr = nullptr;
  }
}
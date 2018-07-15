#include "Runtime.h"
#include "Logger.h"
#include "RuntimeInterface.h"
#include "TypeIO.h"

#include "llvm/ADT/Optional.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace typeart {

std::string TypeArtRT::defaultTypeFileName{"types.yaml"};

template <typename T>
inline const void* addByteOffset(const void* addr, T offset) {
  return static_cast<const void*>(static_cast<const uint8_t*>(addr) + offset);
}

inline static std::string toString(const void* addr, int typeId, size_t count, size_t typeSize, int isLocal) {
  std::stringstream s;
  // clang-format off
  s << addr
    << ". typeId: " << typeId
    << ". count: " << count
    << ". typeSize " << typeSize
    << ". local: " << isLocal;
  // clang-format on
  return s.str();
}

inline static std::string toString(const PointerInfo& info) {
  return toString(info.addr, info.typeId, info.count, info.typeSize, -1);
}

TypeArtRT::TypeArtRT() {
  // Try to load types from specified file first.
  // Then look at default location.
  const char* typeFile = std::getenv("TYPE_FILE");
  if (typeFile) {
    if (!loadTypes(typeFile)) {
      LOG_ERROR("Failed to load recorded types from " << typeFile);
      std::exit(EXIT_FAILURE);  // TODO: Error handling
    }
  } else {
    if (!loadTypes(defaultTypeFileName)) {
      LOG_ERROR("No type file with default name \""
                << defaultTypeFileName
                << "\" in current directory. To specify a different file, edit the TYPE_FILE environment variable.");
      std::exit(EXIT_FAILURE);  // TODO: Error handling
    }
  }

  std::stringstream ss;
  for (auto structInfo : typeDB.getStructList()) {
    ss << structInfo.name << ", ";
  }
  LOG_INFO("Recorded types: " << ss.str());

  stackVars.container().reserve(1024);

  printTraceStart();
}

bool TypeArtRT::loadTypes(const std::string& file) {
  TypeIO cio(typeDB);
  return cio.load(file);
}

void TypeArtRT::printTraceStart() {
  LOG_TRACE("TypeART Runtime Trace");
  LOG_TRACE("**************************");
  LOG_TRACE("Operation  Address   Type   Size   Count  Stack/Heap");
  LOG_TRACE("--------------------------------------------------------");
}

llvm::Optional<PointerInfo> TypeArtRT::findBaseAddress(const void* addr) const {
  if (typeMap.empty() || addr < typeMap.begin()->first) {
    return llvm::None;
  }

  auto it = typeMap.lower_bound(addr);
  if (it == typeMap.end()) {
    // No element bigger than base address
    return {typeMap.rbegin()->second};
  }

  if (it->first == addr) {
    // Exact match
    return {it->second};
  }
  // Base address
  return {std::prev(it)->second};
}

size_t TypeArtRT::getMemberIndex(typeart_struct_layout structInfo, size_t offset) const {
  size_t n = structInfo.len;
  if (offset > structInfo.offsets[n - 1]) {
    return n - 1;
  }

  size_t i = 0;
  while (i < n - 1 && offset >= structInfo.offsets[i + 1]) {
    i++;
  }
  return i;
}

TypeArtRT::TypeArtStatus TypeArtRT::getSubTypeInfo(const void* baseAddr, size_t offset,
                                                   typeart_struct_layout containerInfo, typeart::TypeInfo* subType,
                                                   const void** subTypeBaseAddr, size_t* subTypeOffset,
                                                   size_t* subTypeCount) const {
  if (offset >= containerInfo.extent) {
    return TA_BAD_OFFSET;
  }

  // Get index of the struct member at the address
  size_t memberIndex = getMemberIndex(containerInfo, offset);

  auto memberType = containerInfo.member_types[memberIndex];
  assert((memberType.kind == STRUCT || memberType.kind == BUILTIN || memberType.kind == POINTER) &&
         "Type kind typeart be either STRUCT, BUILTIN or POINTER");

  size_t baseOffset = containerInfo.offsets[memberIndex];
  assert(offset >= baseOffset && "Invalid offset values");

  size_t internalOffset = offset - baseOffset;
  size_t typeSize = typeDB.getTypeSize(memberType);
  size_t offsetInTypeSize = internalOffset / typeSize;
  size_t newOffset = internalOffset % typeSize;

  // If newOffset != 0, the subtype cannot be atomic, i.e. must be a struct
  if (newOffset != 0) {
    if (memberType.kind != STRUCT) {
      return TA_BAD_ALIGNMENT;
    }
  }

  // Ensure that the array index is in bounds
  if (offsetInTypeSize >= containerInfo.count[memberIndex]) {
    // Address points to padding
    return TA_BAD_ALIGNMENT;
  }

  *subType = memberType;
  *subTypeBaseAddr = addByteOffset(baseAddr, baseOffset);
  *subTypeOffset = newOffset;
  *subTypeCount = containerInfo.count[memberIndex] - offsetInTypeSize;

  return TA_OK;
}

TypeArtRT::TypeArtStatus TypeArtRT::getSubTypeInfo(const void* baseAddr, size_t offset,
                                                   const StructTypeInfo& containerInfo, typeart::TypeInfo* subType,
                                                   const void** subTypeBaseAddr, size_t* subTypeOffset,
                                                   size_t* subTypeCount) const {
  typeart_struct_layout structLayout;
  structLayout.id = containerInfo.id;
  structLayout.name = containerInfo.name.c_str();
  structLayout.len = containerInfo.numMembers;
  structLayout.extent = containerInfo.extent;
  structLayout.offsets = &containerInfo.offsets[0];
  structLayout.member_types = &containerInfo.memberTypes[0];
  structLayout.count = &containerInfo.arraySizes[0];
  return getSubTypeInfo(baseAddr, offset, structLayout, subType, subTypeBaseAddr, subTypeOffset, subTypeCount);
}

TypeArtRT::TypeArtStatus TypeArtRT::getTypeInfoInternal(const void* baseAddr, size_t offset,
                                                        const StructTypeInfo& containerInfo, typeart::TypeInfo* type,
                                                        size_t* count) const {
  assert(offset < containerInfo.extent && "Something went wrong with the base address computation");

  TypeArtStatus status;
  TypeInfo subType;
  const void* subTypeBaseAddr;
  size_t subTypeOffset;
  size_t subTypeCount;

  // Resolve type recursively, until the address matches exactly
  do {
    status = getSubTypeInfo(baseAddr, offset, containerInfo, &subType, &subTypeBaseAddr, &subTypeOffset, &subTypeCount);

    if (status != TA_OK) {
      return status;
    }

    baseAddr = subTypeBaseAddr;
    offset = subTypeOffset;
  } while (offset != 0);
  *type = subType;
  *count = subTypeCount;
  return TA_OK;
}

TypeArtRT::TypeArtStatus TypeArtRT::getTypeInfo(const void* addr, typeart::TypeInfo* type, size_t* count) const {
  TypeInfo containingType;
  size_t containingTypeCount;
  const void* containingTypeAddr;
  size_t internalOffset;

  // First, retrieve the containing type
  TypeArtStatus status =
      getContainingTypeInfo(addr, &containingType, &containingTypeCount, &containingTypeAddr, &internalOffset);
  if (status != TA_OK) {
    return status;
  }

  // Check for exact address match
  if (internalOffset == 0) {
    *type = containingType;
    *count = containingTypeCount;
    return TA_OK;
  }

  if (typeDB.isBuiltinType(containingType.id)) {
    // Address points to the middle of a builtin type
    return TA_BAD_ALIGNMENT;
  }

  // Resolve struct recursively
  auto structInfo = typeDB.getStructInfo(containingType.id);
  if (structInfo) {
    return getTypeInfoInternal(containingTypeAddr, internalOffset, *structInfo, type, count);
  }
  return TA_INVALID_ID;
}

TypeArtRT::TypeArtStatus TypeArtRT::getContainingTypeInfo(const void* addr, typeart::TypeInfo* type, size_t* count,
                                                          const void** baseAddress, size_t* offset) const {
  // Find the start address of the containing buffer
  auto ptrData = findBaseAddress(addr);

  if (ptrData) {
    const auto& basePtrInfo = ptrData.getValue();
    auto basePtr = basePtrInfo.addr;

    //    auto basePtrInfo = typeMap.find(basePtr)->second;

    // Check for exact match -> no further checks and offsets calculations needed
    if (basePtr == addr) {
      *type = typeDB.getTypeInfo(basePtrInfo.typeId);
      *count = basePtrInfo.count;
      *baseAddress = addr;
      *offset = 0;
      return TA_OK;
    }

    // The address points inside a known array
    const void* blockEnd = addByteOffset(basePtr, basePtrInfo.count * basePtrInfo.typeSize);

    // Ensure that the given address is in bounds and points to the start of an element
    if (addr >= blockEnd) {
      return TA_UNKNOWN_ADDRESS;
    }

    assert(addr >= basePtr && "Error in base address computation");
    size_t addrDif = reinterpret_cast<size_t>(addr) - reinterpret_cast<size_t>(basePtr);

    // Offset of the pointer w.r.t. the start of the containing type
    size_t internalOffset = addrDif % basePtrInfo.typeSize;

    // Array index
    size_t typeOffset = addrDif / basePtrInfo.typeSize;
    size_t typeCount = basePtrInfo.count - typeOffset;

    // Retrieve and return type information
    // TODO: Ensure that ID is valid
    *type = typeDB.getTypeInfo(basePtrInfo.typeId);
    *count = typeCount;
    *baseAddress = addByteOffset(basePtr, typeOffset * basePtrInfo.typeSize);
    *offset = internalOffset;
    return TA_OK;
  }
  return TA_UNKNOWN_ADDRESS;
}

TypeArtRT::TypeArtStatus TypeArtRT::getBuiltinInfo(const void* addr, typeart::BuiltinType* type) const {
  TypeInfo info;
  size_t count;
  TypeArtStatus result = getTypeInfo(addr, &info, &count);
  if (result == TA_OK) {
    if (info.kind == BUILTIN) {
      *type = static_cast<BuiltinType>(info.id);
      return TA_OK;
    }
    return TA_WRONG_KIND;
  }
  return result;
}

TypeArtRT::TypeArtStatus TypeArtRT::getStructInfo(int id, const StructTypeInfo** structInfo) const {
  TypeInfo typeInfo = typeDB.getTypeInfo(id);
  // Requested ID must correspond to a struct
  if (typeInfo.kind != STRUCT) {
    return TA_WRONG_KIND;
  }

  auto result = typeDB.getStructInfo(id);

  if (result) {
    *structInfo = result;
    return TA_OK;
  }
  return TA_INVALID_ID;
}

const std::string& TypeArtRT::getTypeName(int id) const {
  return typeDB.getTypeName(id);
}

void TypeArtRT::getReturnAddress(const void* addr, const void** retAddr) {
  const void* basePtr = findBaseAddress(addr);

  if (basePtr) {
    auto basePtrInfo = typeMap.find(basePtr)->second;
    *retAddr = basePtrInfo.debug;
  } else {
    *retAddr = nullptr;
  }
}

void TypeArtRT::onAlloc(const void* addr, int typeId, size_t count, size_t typeSize, bool isLocal,
                        const void* retAddr) {
  auto it = typeMap.find(addr);
  if (it != typeMap.end()) {
    const auto info = (*it).second;
    LOG_ERROR("Already exists: " << toString(addr, typeId, count, typeSize, isLocal));
    LOG_ERROR("Data in map is: " << toString(info));
  } else {
    typeMap[addr] = {addr, typeId, count, typeSize, retAddr};
    auto typeString = typeDB.getTypeName(typeId);
    LOG_TRACE("Alloc " << addr << " " << typeString << " " << typeSize << " " << count << " " << (isLocal ? "S" : "H"));
    if (isLocal) {
      //      LOG_TRACE("Alloc is stack " << stackVars.size() << " " << stackVars.container().size());
      stackVars.push_back(addr);
    }
  }
}

void TypeArtRT::onFree(const void* addr) {
  auto it = typeMap.find(addr);
  if (it != typeMap.end()) {
    LOG_TRACE("Free " << toString((*it).second));
    typeMap.erase(it);
  } else {
    LOG_ERROR("Free recorded on unregistered address: " << addr);
    // TODO: What to do when not found?
  }
}

void TypeArtRT::onLeaveScope(size_t alloca_count) {
  if (alloca_count > stackVars.size()) {
    LOG_ERROR("Stack is smaller than requested de-allocation count. alloca_count: " << alloca_count
                                                                                    << ". size: " << stackVars.size());
    alloca_count = stackVars.size();
  }

  const auto cend = stackVars.cend();
  const auto start_pos = (cend - alloca_count);
  LOG_TRACE("Freeing stack (" << alloca_count << ")  " << std::distance(start_pos, stackVars.cend()))
  std::for_each(start_pos, cend, [&](const void* addr) { onFree(addr); });
  stackVars.free(alloca_count);
  LOG_TRACE("Stack after free: " << stackVars.size());

  // FIXME this is an expensive O(n) operation due to using a vector for stackBars,
  // and is strictly speaking, not a necessary operation.
  // Possible fix: use an additional index and set it to the "end" of valid addresses in the vector
  // for push_back we need to check if the index is then outside of the vector range and use either push_back or
  // vecAdr[index]
  // stackVars.erase(start_pos, cend);
}

}  // namespace typeart

void __typeart_alloc(void* addr, int typeId, size_t count, size_t typeSize, int isLocal) {
  const void* ret_adr = __builtin_return_address(0);
  typeart::TypeArtRT::get().onAlloc(addr, typeId, count, typeSize, isLocal, ret_adr);
}

void __typeart_free(void* addr) {
  //  const void* ret_adr = __builtin_return_address(0);
  typeart::TypeArtRT::get().onFree(addr);
}

void __typeart_leave_scope(size_t alloca_count) {
  //  const void* ret_adr = __builtin_return_address(0);
  typeart::TypeArtRT::get().onLeaveScope(alloca_count);
}

typeart_status typeart_get_builtin_type(const void* addr, typeart::BuiltinType* type) {
  return typeart::TypeArtRT::get().getBuiltinInfo(addr, type);
}

typeart_status typeart_get_type(const void* addr, typeart::TypeInfo* type, size_t* count) {
  return typeart::TypeArtRT::get().getTypeInfo(addr, type, count);
}

typeart_status typeart_get_containing_type(const void* addr, typeart::TypeInfo* type, size_t* count,
                                           const void** base_address, size_t* offset) {
  return typeart::TypeArtRT::get().getContainingTypeInfo(addr, type, count, base_address, offset);
}

typeart_status typeart_get_subtype(const void* base_addr, size_t offset, typeart_struct_layout container_layout,
                                   typeart::TypeInfo* subtype, const void** subtype_base_addr, size_t* subtype_offset,
                                   size_t* subtype_count) {
  return typeart::TypeArtRT::get().getSubTypeInfo(base_addr, offset, container_layout, subtype, subtype_base_addr,
                                                  subtype_offset, subtype_count);
}

typeart_status typeart_resolve_type(int id, typeart_struct_layout* struct_layout) {
  const typeart::StructTypeInfo* structInfo;
  typeart_status status = typeart::TypeArtRT::get().getStructInfo(id, &structInfo);
  if (status == TA_OK) {
    struct_layout->id = structInfo->id;
    struct_layout->name = structInfo->name.c_str();
    struct_layout->len = structInfo->numMembers;
    struct_layout->extent = structInfo->extent;
    struct_layout->offsets = &structInfo->offsets[0];
    struct_layout->member_types = &structInfo->memberTypes[0];
    struct_layout->count = &structInfo->arraySizes[0];
    return TA_OK;
  }
  return status;
}

const char* typeart_get_type_name(int id) {
  return typeart::TypeArtRT::get().getTypeName(id).c_str();
}

void typeart_get_return_address(const void* addr, const void** retAddr) {
  return typeart::TypeArtRT::get().getReturnAddress(addr, retAddr);
}

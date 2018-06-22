#include "Runtime.h"
#include "RuntimeInterface.h"
#include "RuntimeUtil.h"
#include "TypeIO.h"
#include "support/Logger.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>

namespace typeart {

std::string TypeArtRT::defaultTypeFileName{"types.yaml"};

TypeArtRT::TypeArtRT() {

  // Try to load types from specified file first.
  // Then look at default location.
  const char* typeFile = std::getenv("TYPE_FILE");
  if (typeFile) {
      if (!loadTypes(typeFile)) {
          LOG_ERROR("Failed to load recorded types from " << typeFile);
          exit(0); // TODO: Error handling
      }
  } else {
      if (!loadTypes(defaultTypeFileName)) {
          LOG_ERROR(
                  "No type file with default name \"" << defaultTypeFileName << "\" in current directory. To specify a different file, edit the TYPE_FILE environment variable.");
          exit(0); // TODO: Error handling
      }
  }

  std::stringstream ss;
  for (auto structInfo : typeDB.getStructList()) {
    ss << structInfo.name << ", ";
  }
  LOG_INFO("Recorded types: " << ss.str());

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

const void* TypeArtRT::findBaseAddress(const void* addr) const {
  if (typeMap.empty() || addr < typeMap.begin()->first) {
    return nullptr;
  }

  auto it = typeMap.lower_bound(addr);
  if (it == typeMap.end()) {
    // No element bigger than base address
    return typeMap.rbegin()->first;
  }

  if (it->first == addr) {
    // Exact match
    return it->first;
  }
  // Base address
  return (std::prev(it))->first;
}

size_t TypeArtRT::getMemberIndex(const StructTypeInfo& structInfo, size_t offset) const {
  if (offset > structInfo.offsets.back()) {
    return structInfo.numMembers - 1;
  }

  size_t i = 0;
  while (i < structInfo.numMembers - 1 && offset >= structInfo.offsets[i + 1]) {
    i++;
  }
  return i;
}

TypeArtRT::TypeArtStatus TypeArtRT::getTypeInfoInternal(const void* baseAddr, size_t offset, const StructTypeInfo& containerInfo,
                                             typeart::TypeInfo* type, size_t* count) const {
  assert(offset < containerInfo.extent && "Something went wrong with the base address computation");

  // std::cout << "internal: base=" << baseAddr << ", offset=" << offset << ", container=" << containerInfo.name <<
  // std::endl;

  size_t memberIndex = getMemberIndex(containerInfo, offset);

  auto memberType = containerInfo.memberTypes[memberIndex];
  size_t baseOffset = containerInfo.offsets[memberIndex];

  assert(offset >= baseOffset && "Invalid offset values");

  size_t internalOffset = offset - baseOffset;

  assert((memberType.kind == STRUCT || memberType.kind == BUILTIN || memberType.kind == POINTER) &&
         "Type kind typeart be either STRUCT, BUILTIN or POINTER");

  size_t typeSize = typeDB.getTypeSize(memberType);
  if (memberType.kind == STRUCT && internalOffset % typeSize != 0) {
    // Address points inside of a sub-struct -> we have to go deeper
    auto memberStructInfo = typeDB.getStructInfo(memberType.id);
    if (memberStructInfo) {
      const void* newBaseAddr = addByteOffset(baseAddr, baseOffset);
      return getTypeInfoInternal(newBaseAddr, internalOffset, *memberStructInfo, type, count);
    }
    return INVALID_ID;
  }

  // Type is either atomic or the address corresponds to the start of a sub-struct - offset must match up with type size
  if (internalOffset % typeSize == 0) {
    size_t offsetInTypeSize = internalOffset / typeSize;
    if (offsetInTypeSize >= containerInfo.arraySizes[memberIndex]) {
      // Address points to padding
      return BAD_ALIGNMENT;
    }
    *type = memberType;
    *count = containerInfo.arraySizes[memberIndex] - offsetInTypeSize;
    return SUCCESS;
  }
  return BAD_ALIGNMENT;
}

TypeArtRT::TypeArtStatus TypeArtRT::getTypeInfo(const void* addr, typeart::TypeInfo* type, size_t* count) const {
  const void* basePtr = findBaseAddress(addr);

  if (basePtr) {
    auto basePtrInfo = typeMap.find(basePtr)->second;

    // Check for exact match
    if (basePtr == addr) {
      *type = typeDB.getTypeInfo(basePtrInfo.typeId);
      *count = basePtrInfo.count;
      return SUCCESS;
    }

    // The address points inside a known array
    const void* blockEnd = addByteOffset(basePtr, basePtrInfo.count * basePtrInfo.typeSize);
    // Ensure that the given address is in bounds and points to the start of an element
    if (addr >= blockEnd) {
      return UNKNOWN_ADDRESS;
    }
    size_t addrDif = reinterpret_cast<size_t>(addr) - reinterpret_cast<size_t>(basePtr);
    size_t internalOffset =
        addrDif % basePtrInfo.typeSize;  // Offset of the pointer w.r.t. the start of the containing type
    if (internalOffset != 0) {
      if (typeDB.isBuiltinType(basePtrInfo.typeId)) {
        return BAD_ALIGNMENT;  // Address points to the middle of a builtin type
      } else {
        assert(addrDif >= internalOffset && "Error in computation of offset within struct");
        // Compute the address of the start of the struct
        const void* structAddr = addByteOffset(basePtr, addrDif - internalOffset);
        auto structInfo = typeDB.getStructInfo(basePtrInfo.typeId);
        if (structInfo) {
          return getTypeInfoInternal(structAddr, internalOffset, *structInfo, type, count);
        }
        return INVALID_ID;
      }
    } else {
      // Compute the element count from the given address
      *count = basePtrInfo.count - addrDif / basePtrInfo.typeSize;
      *type = typeDB.getTypeInfo(basePtrInfo.typeId);
      return SUCCESS;
    }
  }

  return UNKNOWN_ADDRESS;
}

TypeArtRT::TypeArtStatus TypeArtRT::getBuiltinInfo(const void* addr, typeart::BuiltinType* type) const {
  TypeInfo info;
  size_t count;
  TypeArtStatus result = getTypeInfo(addr, &info, &count);
  if (result == SUCCESS) {
    if (info.kind == BUILTIN) {
      *type = static_cast<BuiltinType>(info.id);
      return SUCCESS;
    }
    return WRONG_KIND;
  }
  return result;
}

TypeArtRT::TypeArtStatus TypeArtRT::getStructInfo(int id, const StructTypeInfo** structInfo) const {
  TypeInfo typeInfo = typeDB.getTypeInfo(id);
  // Requested ID must correspond to a struct
  if (typeInfo.kind != STRUCT) {
    return WRONG_KIND;
  }

  auto result = typeDB.getStructInfo(id);

  if (result) {
    *structInfo = result;
    return SUCCESS;
  }
  return INVALID_ID;
}

const std::string& TypeArtRT::getTypeName(int id) const {
  return typeDB.getTypeName(id);
}

void TypeArtRT::onAlloc(const void* addr, int typeId, size_t count, size_t typeSize, bool isLocal) {
  auto it = typeMap.find(addr);
  if (it != typeMap.end()) {
    LOG_ERROR("Alloc recorded with unknown type ID: " << typeId);
    // TODO: What should the behaviour be here?
  } else {
    typeMap[addr] = {addr, typeId, count, typeSize};
    auto typeString = typeDB.getTypeName(typeId);
    LOG_TRACE("Alloc " << addr << " " << typeString << " " << typeSize << " " << count << " " << (isLocal ? "S" : "H"));
    if (isLocal) {
      if (scopes.empty()) {
        LOG_ERROR("Error while recording stack allocation: no active scope");
        return;
      }
      auto& scope = scopes.back();
      scope.push_back(addr);
    }
  }
}

void TypeArtRT::onFree(const void* addr) {
  auto it = typeMap.find(addr);
  if (it != typeMap.end()) {
    typeMap.erase(it);
    LOG_TRACE("Free " << addr);
  } else {
    LOG_ERROR("Free recorded on unregistered address: " << addr);
    // TODO: What to do when not found?
  }
}

void TypeArtRT::onEnterScope() {
  LOG_TRACE("Entering scope");
  scopes.push_back({});
}

void TypeArtRT::onLeaveScope() {
  if (scopes.empty()) {
    LOG_ERROR("Error while leaving scope: no active scope");
    return;
  }
  std::vector<const void*>& scope = scopes.back();
  LOG_TRACE("Leaving scope: freeing " << scope.size() << " stack entries");
  for (const void* addr : scope) {
    onFree(addr);
  }
  scopes.pop_back();
}

}  // namespace typeart

void __typeart_alloc(void* addr, int typeId, size_t count, size_t typeSize, int isLocal) {
  typeart::TypeArtRT::get().onAlloc(addr, typeId, count, typeSize, isLocal);
}

void __typeart_free(void* addr) {
  typeart::TypeArtRT::get().onFree(addr);
}

void __typeart_enter_scope() {
  typeart::TypeArtRT::get().onEnterScope();
}

void __typeart_leave_scope() {
  typeart::TypeArtRT::get().onLeaveScope();
}

typeart_status typeart_get_builtin_type(const void* addr, typeart::BuiltinType* type) {
  return typeart::TypeArtRT::get().getBuiltinInfo(addr, type);
}

typeart_status typeart_get_type(const void* addr, typeart::TypeInfo* type, size_t* count) {
  return typeart::TypeArtRT::get().getTypeInfo(addr, type, count);
}

typeart_status typeart_resolve_type(int id, size_t* len, const typeart_type_info** types, const size_t** count,
                                   const size_t** offsets, size_t* extent) {
  const typeart::StructTypeInfo* structInfo;
  typeart_status status = typeart::TypeArtRT::get().getStructInfo(id, &structInfo);
  if (status == SUCCESS) {
    size_t n = structInfo->numMembers;
    *len = n;
    *types = &structInfo->memberTypes[0];
    *count = &structInfo->arraySizes[0];
    *offsets = &structInfo->offsets[0];
    *extent = structInfo->extent;
    return SUCCESS;
  }
  return status;
}

const char* typeart_get_type_name(int id) {
  return typeart::TypeArtRT::get().getTypeName(id).c_str();
}

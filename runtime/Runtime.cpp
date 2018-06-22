#include "Runtime.h"
#include "RuntimeInterface.h"

#include <TypeIO.h>
//#include <TypeDB.h>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
// TODO: Think about putting the logger somewhere else
#include "../lib/support/Logger.h"

namespace typeart {

TypeArtRT::TypeArtRT() {
  std::string typeFile = std::string("./") + typeFileName;
  if (!loadTypes(typeFile)) {
    // Check env
    const char* dir = std::getenv("TYPE_PATH");  // TODO: Name
    if (dir) {
      typeFile = std::string(dir) + "/" + typeFileName;
    } else {
      LOG_ERROR(
          "No type file in current directory. To specify a different path, edit the TYPE_PATH environment variable.");
    }
    if (!loadTypes(typeFile)) {
      LOG_ERROR("Failed to load recorded types from " << typeFile);
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
  if (addr < typeMap.begin()->first) {
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

size_t TypeArtRT::getMemberIndex(const StructTypeInfo &structInfo, size_t offset) const {

  if (offset > structInfo.offsets.back()) {
    return structInfo.numMembers - 1;
  }

  size_t i = 0;
  while (i < structInfo.numMembers - 1 && offset >= structInfo.offsets[i + 1]) {
    i++;
  }
  return i;
}

lookup_result TypeArtRT::getTypeInfoInternal(const void* baseAddr, size_t offset, const StructTypeInfo& containerInfo,
                                             typeart::TypeInfo* type, size_t* count) const {
  assert(offset < containerInfo.extent && "Something went wrong with the base address computation");

  // std::cout << "internal: base=" << baseAddr << ", offset=" << offset << ", container=" << containerInfo.name <<
  // std::endl;

  size_t memberIndex = getMemberIndex(containerInfo, offset);

  auto memberType = containerInfo.memberTypes[memberIndex];
  size_t baseOffset = containerInfo.offsets[memberIndex];
  size_t internalOffset = offset - baseOffset;

  assert(offset >= baseOffset && "Invalid offset values");

  assert((memberType.kind == STRUCT || memberType.kind == BUILTIN || memberType.kind == POINTER) &&
         "Type kind typeart be either STRUCT, BUILTIN or POINTER");

  size_t typeSize = 0;
  if (memberType.kind == STRUCT) {
    auto memberStructInfo = typeDB.getStructInfo(memberType.id);
    typeSize = memberStructInfo->extent;
    if (internalOffset % typeSize != 0) {
      // Address points inside of a sub-struct -> we have to go deeper
      const void* newBaseAddr = static_cast<const void*>(static_cast<const uint8_t*>(baseAddr) + baseOffset);
      return getTypeInfoInternal(newBaseAddr, internalOffset, *memberStructInfo, type, count);
    }
  } else if (memberType.kind == POINTER) {
    typeSize = sizeof(void*);
  } else {
    typeSize = typeDB.getBuiltinTypeSize(memberType.id);
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

lookup_result TypeArtRT::getTypeInfo(const void* addr, typeart::TypeInfo* type, size_t* count) const {
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
    const void* blockEnd =
        static_cast<const void*>(static_cast<const uint8_t*>(basePtr) + basePtrInfo.count * basePtrInfo.typeSize);
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
        const void* structAddr =
            static_cast<const void*>(static_cast<const uint8_t*>(basePtr) + addrDif - internalOffset);
        auto structInfo = typeDB.getStructInfo(basePtrInfo.typeId);

        auto result = getTypeInfoInternal(structAddr, internalOffset, *structInfo, type, count);
        // *count = basePtrInfo.count - addrDif / basePtrInfo.typeSize;  // TODO: Correct behavior?
        return result;
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

lookup_result TypeArtRT::getBuiltinInfo(const void* addr, typeart::BuiltinType* type) const {
  TypeInfo info;
  size_t count;
  lookup_result result = getTypeInfo(addr, &info, &count);
  if (result == SUCCESS) {
    if (info.kind == BUILTIN) {
      *type = static_cast<BuiltinType>(info.id);
      return SUCCESS;
    }
    return WRONG_KIND;
  }
  return result;
}

lookup_result TypeArtRT::getStructInfo(int id, const StructTypeInfo** structInfo) const {
  TypeInfo typeInfo = typeDB.getTypeInfo(id);
  // Requested ID typeart correspond to a struct
  if (typeInfo.kind != STRUCT) {
    return WRONG_KIND;
  }

  *structInfo = typeDB.getStructInfo(id);

  return SUCCESS;
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

lookup_result typeart_get_builtin_type(const void* addr, typeart::BuiltinType* type) {
  return typeart::TypeArtRT::get().getBuiltinInfo(addr, type);
}

lookup_result typeart_get_type(const void* addr, typeart::TypeInfo* type, size_t* count) {
  return typeart::TypeArtRT::get().getTypeInfo(addr, type, count);
}

lookup_result typeart_resolve_type(int id, size_t* len, const typeart_type_info** types, const size_t** count,
                                   const size_t** offsets, size_t* extent) {
  const typeart::StructTypeInfo* structInfo;
  lookup_result status = typeart::TypeArtRT::get().getStructInfo(id, &structInfo);
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

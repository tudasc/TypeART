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
  LOG_TRACE("Operation  Address   Type   Size   Count");
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

lookup_result TypeArtRT::getTypeInfoInternal(const void* baseAddr, size_t offset, const StructTypeInfo& containerInfo,
                                             typeart::TypeInfo* type, size_t* count) const {
  // std::cerr << "Type is " << containerInfo.name << std::endl;

  assert(offset < containerInfo.extent && "Something went wrong with the base address computation");
  // std::cout << "internal: base=" << baseAddr << ", offset=" << offset << ", container=" << containerInfo.name <<
  // std::endl;
  size_t i = 0;
  // This should always be 0, but safety doesn't hurt
  size_t baseOffset = containerInfo.offsets.front();

  if (offset > containerInfo.offsets.back()) {
    i = containerInfo.numMembers - 1;
    baseOffset = containerInfo.offsets.back();
  } else {
    while (i < containerInfo.numMembers - 1 && offset >= containerInfo.offsets[i + 1]) {
      i++;
      baseOffset = containerInfo.offsets[i];
    }
  }

  auto memberInfo = containerInfo.memberTypes[i];

  // Offset corresponds directly to a member
  if (baseOffset == offset) {
    *type = memberInfo;
    *count = containerInfo.arraySizes[i];
    return SUCCESS;
  }

  // Offset corresponds to location inside the member

  assert((offset > containerInfo.offsets[i] &&
          (i == containerInfo.numMembers - 1 || baseOffset < containerInfo.offsets[i + 1])) &&
         "Bug in offset computation code");

  if (memberInfo.kind == STRUCT) {
    // Address points inside of a sub-struct -> we have to go deeper
    auto memberStructInfo = typeDB.getStructInfo(memberInfo.id);
    const void* newBaseAddr = static_cast<const void*>(static_cast<const uint8_t*>(baseAddr) + baseOffset);
    size_t newOffset = offset - baseOffset;
    return getTypeInfoInternal(newBaseAddr, newOffset, *memberStructInfo, type, count);
  } else {
    assert((memberInfo.kind == BUILTIN || memberInfo.kind == POINTER) &&
           "Type kind typeart be either STRUCT, BUILTIN or POINTER");
    // Assume type is pointer by default
    int typeSize = sizeof(void*);
    if (memberInfo.kind == BUILTIN) {
      // Fetch actual size
      typeSize = typeDB.getBuiltinTypeSize(memberInfo.id);
    }

    size_t dif = offset - baseOffset;
    // Type is atomic - offset typeart match up with type size
    if (dif % typeSize == 0) {
      size_t offsetInTypeSize = dif / typeSize;
      if (offsetInTypeSize >= containerInfo.arraySizes[i]) {
        // Address points to padding
        return BAD_ALIGNMENT;
      }
      *type = memberInfo;
      *count = containerInfo.arraySizes[i] - offsetInTypeSize;
      return SUCCESS;
    }
    return BAD_ALIGNMENT;
  }
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

void TypeArtRT::onAlloc(void* addr, int typeId, size_t count, size_t typeSize) {
  auto it = typeMap.find(addr);
  if (it != typeMap.end()) {
    LOG_ERROR("Alloc recorded with unknown type ID: " << typeId);
    // TODO: What should the behaviour be here?
  } else {
    typeMap[addr] = {addr, typeId, count, typeSize};
    auto typeString = typeDB.getTypeName(typeId);
    LOG_TRACE("Alloc " << addr << " " << typeString << " " << typeSize << " " << count);
  }
}

void TypeArtRT::onFree(void* addr) {
  auto it = typeMap.find(addr);
  if (it != typeMap.end()) {
    typeMap.erase(it);
    LOG_TRACE("Free " << addr);
  } else {
    LOG_ERROR("Free recorded on unregistered address: " << addr);
    // TODO: What to do when not found?
  }
}

}  // namespace typeart

void __typeart_alloc(void *addr, int typeId, size_t count, size_t typeSize) {
  typeart::TypeArtRT::get().onAlloc(addr, typeId, count, typeSize);
}

void __typeart_free(void *addr) {
  typeart::TypeArtRT::get().onFree(addr);
}

lookup_result typeart_support_get_builtin_type(const void* addr, typeart::BuiltinType* type) {
  return typeart::TypeArtRT::get().getBuiltinInfo(addr, type);
}

lookup_result typeart_support_get_type(const void* addr, typeart::TypeInfo* type, size_t* count) {
  return typeart::TypeArtRT::get().getTypeInfo(addr, type, count);
}

lookup_result typeart_support_resolve_type(int id, size_t* len, const typeart::TypeInfo* types[], const size_t* count[],
                                           const size_t* offsets[], size_t* extent) {
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

const char* typeart_support_get_type_name(int id) {
  return typeart::TypeArtRT::get().getTypeName(id).c_str();
}

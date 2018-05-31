#include "Runtime.h"
#include "RuntimeInterface.h"

#include <TypeIO.h>
//#include <TypeDB.h>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <algorithm>

void __must_support_alloc(void* addr, int typeId, long count, long typeSize) {
  must::MustSupportRT::get().onAlloc(addr, typeId, count, typeSize);
}

void __must_support_free(void* addr) {
  must::MustSupportRT::get().onFree(addr);
}

lookup_result must_support_get_builtin_type(const void* addr, must::BuiltinType* type) {
  return must::MustSupportRT::get().getBuiltinInfo(addr, type);
}

lookup_result must_support_get_type(const void* addr, must::TypeInfo* type, int* count) {
  return must::MustSupportRT::get().getTypeInfo(addr, type, count);
}

lookup_result must_support_resolve_type(int id, int* len, const must::TypeInfo* types[], const int* count[],
                                        const int* offsets[], int* extent) {
  const must::StructTypeInfo* structInfo;
  lookup_result status = must::MustSupportRT::get().getStructInfo(id, &structInfo);
  if (status == SUCCESS) {
    int n = structInfo->numMembers;
    *len = n;
    *types = &structInfo->memberTypes[0];
    *count = &structInfo->arraySizes[0];
    *offsets = &structInfo->offsets[0];
    *extent = structInfo->extent;
    return SUCCESS;
  }
  return status;
}

const char* must_support_get_type_name(int id) {
  return must::MustSupportRT::get().getTypeName(id).c_str();
}

namespace must {

MustSupportRT::MustSupportRT() {
  std::string typeFile = std::string("./") + configFileName;
  if (!loadTypes(typeFile)) {
    // Check env
    const char* dir = std::getenv("TYPE_PATH"); // TODO: Name
    if (dir) {
        typeFile = std::string(dir) + "/" + configFileName;
    } else {
      std::cerr << "No type file in current directory. To specify a different path, edit the TYPE_PATH environment variable." << std::endl;
    }
    if (!loadTypes(typeFile)) {
      std::cerr << "Failed to load recorded types from " << typeFile << std::endl;
    }
  }

  std::cout << "Recorded types: ";
  for (auto structInfo : typeConfig.getStructList()) {
    std::cout << structInfo.name << ", ";
  }
  std::cout << std::endl;

  printTraceStart();
}

bool MustSupportRT::loadTypes(const std::string& file)
{
  TypeIO cio(&typeConfig);
  return cio.load(file);
}

void MustSupportRT::printTraceStart() {
  std::cout << "MUST Support Runtime Trace" << std::endl;
  std::cout << "**************************" << std::endl;
  std::cout << "Operation  Address   Type   Size   Count" << std::endl;
  std::cout << "--------------------------------------------------------" << std::endl;
}

const void* MustSupportRT::findBaseAddress(const void* addr) const {

  if (addr < typeMap.begin()->first) {
    return nullptr;
  }
  auto it = typeMap.lower_bound(addr);
  if (it == typeMap.end()) {
    return nullptr;
  }

  if (it->first == addr) {
    // Exact match
    return it->first;
  }
  // Base address
  return (std::prev(it))->first;

}

lookup_result MustSupportRT::getTypeInfoInternal(const void* baseAddr, int offset, const StructTypeInfo& containerInfo,
                                                 must::TypeInfo* type) const {
  // std::cerr << "Type is " << containerInfo.name << std::endl;

  assert(offset < containerInfo.extent && "Something went wrong with the base address computation");
  //std::cout << "internal: base=" << baseAddr << ", offset=" << offset << ", container=" << containerInfo.name << std::endl;
  int i = 0;
  // This should always be 0, but safety doesn't hurt
  int baseOffset = containerInfo.offsets.front();

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
    return SUCCESS;
  }

  // Offset corresponds to location inside the member

  assert((offset > containerInfo.offsets[i] &&
          (i == containerInfo.numMembers - 1 || baseOffset < containerInfo.offsets[i + 1])) &&
         "Bug in offset computation code");

  if (memberInfo.kind == STRUCT) {
    // Address points inside of a sub-struct -> we have to go deeper
    auto memberStructInfo = typeConfig.getStructInfo(memberInfo.id);
    return getTypeInfoInternal(baseAddr + baseOffset, offset - baseOffset, *memberStructInfo, type);
  } else {
    assert((memberInfo.kind == BUILTIN || memberInfo.kind == POINTER) &&
           "Type kind must be either STRUCT, BUILTIN or POINTER");
    // Assume type is pointer by default
    int typeSize = sizeof(void*);
    if (memberInfo.kind == BUILTIN) {
      // Fetch actual size
      typeSize = typeConfig.getBuiltinTypeSize(memberInfo.id);
    }

    int dif = offset - baseOffset;
    // Type is atomic - offset must match up with type size
    if (dif % typeSize == 0) {
      if (dif / typeSize >= containerInfo.arraySizes[i]) {
        // Address points to padding
        return BAD_ALIGNMENT;
      }
      *type = memberInfo;
      return SUCCESS;
    }
    return BAD_ALIGNMENT;
  }
}

lookup_result MustSupportRT::getTypeInfo(const void* addr, must::TypeInfo* type, int* count) const {

  const void* basePtr = findBaseAddress(addr);

  if (basePtr) {

    auto basePtrInfo = typeMap.find(basePtr)->second;

    // Check for exact match
    if (basePtr == addr) {
      *type = typeConfig.getTypeInfo(basePtrInfo.typeId);
      *count = basePtrInfo.count;
      return SUCCESS;
    }

    // The address points inside a known array
    const void* blockEnd = basePtr + basePtrInfo.count * basePtrInfo.typeSize;
    // Ensure that the given address is in bounds and points to the start of an element
    if (addr >= blockEnd) {
      return UNKNOWN_ADDRESS;
    }
    long addrDif = (long)addr - (long)basePtr;
    int internalOffset =
        addrDif % basePtrInfo.typeSize;  // Offset of the pointer w.r.t. the start of the containing type
    if (internalOffset != 0) {
      if (typeConfig.isBuiltinType(basePtrInfo.typeId)) {
        return BAD_ALIGNMENT;  // Address points to the middle of a builtin type
      } else {
        const void* structAddr = basePtr + addrDif - internalOffset;
        auto structInfo = typeConfig.getStructInfo(basePtrInfo.typeId);

        auto result = getTypeInfoInternal(structAddr, internalOffset, *structInfo, type);
        *count = basePtrInfo.count - addrDif / basePtrInfo.typeSize; // TODO: Correct behavior?
        return result;
      }
    } else {
      // Compute the element count from the given address
      *count = basePtrInfo.count - addrDif / basePtrInfo.typeSize;
      *type = typeConfig.getTypeInfo(basePtrInfo.typeId);
      return SUCCESS;
    }
  }

  return UNKNOWN_ADDRESS;
}

lookup_result MustSupportRT::getBuiltinInfo(const void* addr, must::BuiltinType* type) const {
  TypeInfo info;
  int count;
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

lookup_result MustSupportRT::getStructInfo(int id, const StructTypeInfo** structInfo) const {
  TypeInfo typeInfo = typeConfig.getTypeInfo(id);
  // Requested ID must correspond to a struct
  if (typeInfo.kind != STRUCT) {
    return WRONG_KIND;
  }

  *structInfo = typeConfig.getStructInfo(id);

  return SUCCESS;
}

const std::string& MustSupportRT::getTypeName(int id) const {
  return typeConfig.getTypeName(id);
}

void MustSupportRT::onAlloc(void* addr, int typeId, long count, long typeSize) {
  auto it = typeMap.find(addr);
  if (it != typeMap.end()) {
    // TODO: What should the behaviour be here?
  } else {
    typeMap[addr] = {addr, typeId, count, typeSize};
    auto typeString = typeConfig.getTypeName(typeId);
    std::cout << "Alloc    " << addr << "   " << typeString << "   " << typeSize << "     " << count << std::endl;
  }
}

void MustSupportRT::onFree(void* addr) {
  auto it = typeMap.find(addr);
  if (it != typeMap.end()) {
    typeMap.erase(it);
    std::cout << "Free     " << addr << std::endl;
  } else {
    // TODO: What to do when not found?
  }
}
}
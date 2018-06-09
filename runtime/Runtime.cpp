#include "Runtime.h"
#include "RuntimeInterface.h"

#include <TypeIO.h>
//#include <TypeDB.h>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace must {

MustSupportRT::MustSupportRT() {
  std::string typeFile = std::string("./") + typeFileName;
  if (!loadTypes(typeFile)) {
    // Check env
    const char* dir = std::getenv("TYPE_PATH");  // TODO: Name
    if (dir) {
      typeFile = std::string(dir) + "/" + typeFileName;
    } else {
      std::cerr
          << "No type file in current directory. To specify a different path, edit the TYPE_PATH environment variable."
          << std::endl;
    }
    if (!loadTypes(typeFile)) {
      std::cerr << "Failed to load recorded types from " << typeFile << std::endl;
    }
  }

  std::cout << "Recorded types: ";
  for (auto structInfo : typeDB.getStructList()) {
    std::cout << structInfo.name << ", ";
  }
  std::cout << std::endl;

  printTraceStart();
}

bool MustSupportRT::loadTypes(const std::string& file) {
  TypeIO cio(typeDB);
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

lookup_result MustSupportRT::getTypeInfoInternal(const void* baseAddr, size_t offset,
                                                 const StructTypeInfo& containerInfo, must::TypeInfo* type, size_t* count) const {
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
           "Type kind must be either STRUCT, BUILTIN or POINTER");
    // Assume type is pointer by default
    int typeSize = sizeof(void*);
    if (memberInfo.kind == BUILTIN) {
      // Fetch actual size
      typeSize = typeDB.getBuiltinTypeSize(memberInfo.id);
    }

    size_t dif = offset - baseOffset;
    // Type is atomic - offset must match up with type size
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

lookup_result MustSupportRT::getTypeInfo(const void* addr, must::TypeInfo* type, size_t* count) const {
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

lookup_result MustSupportRT::getBuiltinInfo(const void* addr, must::BuiltinType* type) const {
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

lookup_result MustSupportRT::getStructInfo(int id, const StructTypeInfo** structInfo) const {
  TypeInfo typeInfo = typeDB.getTypeInfo(id);
  // Requested ID must correspond to a struct
  if (typeInfo.kind != STRUCT) {
    return WRONG_KIND;
  }

  *structInfo = typeDB.getStructInfo(id);

  return SUCCESS;
}

const std::string& MustSupportRT::getTypeName(int id) const {
  return typeDB.getTypeName(id);
}

void MustSupportRT::onAlloc(void* addr, int typeId, size_t count, size_t typeSize) {
  auto it = typeMap.find(addr);
  if (it != typeMap.end()) {
    // TODO: What should the behaviour be here?
  } else {
    typeMap[addr] = {addr, typeId, count, typeSize};
    auto typeString = typeDB.getTypeName(typeId);
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

}  // namespace must

void __must_support_alloc(void* addr, int typeId, size_t count, size_t typeSize) {
  must::MustSupportRT::get().onAlloc(addr, typeId, count, typeSize);
}

void __must_support_free(void* addr) {
  must::MustSupportRT::get().onFree(addr);
}

lookup_result must_support_get_builtin_type(const void* addr, must::BuiltinType* type) {
  return must::MustSupportRT::get().getBuiltinInfo(addr, type);
}

lookup_result must_support_get_type(const void* addr, must::TypeInfo* type, size_t* count) {
  return must::MustSupportRT::get().getTypeInfo(addr, type, count);
}

lookup_result must_support_resolve_type(int id, size_t* len, const must::TypeInfo* types[], const size_t* count[],
                                        const size_t* offsets[], size_t* extent) {
  const must::StructTypeInfo* structInfo;
  lookup_result status = must::MustSupportRT::get().getStructInfo(id, &structInfo);
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

const char* must_support_get_type_name(int id) {
  return must::MustSupportRT::get().getTypeName(id).c_str();
}

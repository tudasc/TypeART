#include "Runtime.h"
#include "RuntimeInterface.h"

#include <ConfigIO.h>
//#include <TypeConfig.h>
#include <iostream>
#include <assert.h>

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
  std::cout << "Looking up type at address " << addr << std::endl;
  return must::MustSupportRT::get().getTypeInfo(addr, type, count);
}

lookup_result must_support_resolve_type(int id, int* len, must::TypeInfo* types[], int* count[], int* offsets[],
                                        int* extent) {
  return must::MustSupportRT::get().resolveType(id, len, types, count, offsets, extent);
}

const char* must_support_get_type_name(int id) {
  return must::MustSupportRT::get().getTypeName(id).c_str();
}

namespace must {

MustSupportRT::MustSupportRT() {
  ConfigIO cio(&typeConfig);
  cio.load("/tmp/musttypes");
  std::cout << "Recorded types: ";
  for (auto structInfo : typeConfig.getStructList()) {
    std::cout << structInfo.name << ", ";
  }
  std::cout << std::endl;
  printTraceStart();
}

void MustSupportRT::printTraceStart() {
  std::cout << "MUST Support Runtime Trace" << std::endl;
  std::cout << "**************************" << std::endl;
  std::cout << "Operation  Address   Type   Size   Count" << std::endl;
  std::cout << "--------------------------------------------------------" << std::endl;
}

// const PointerInfo* MustSupportRT::getPtrInfo(void *ptr) const {
//  auto it = typeMap.find(ptr);
//  if (it != typeMap.end()) {
//    return &it->second;
//  }
//
//  // TODO: More efficient lookup?
//  // Find possible base pointer
//  const PointerInfo* basePtrInfo = nullptr;
//  void* basePtr = 0;
//  for (it = typeMap.begin(); it != typeMap.end(); it++) {
//    if (it->first < ptr && it->first > basePtr) {
//      basePtr = it->first;
//      basePtrInfo = &it->second;
//    }
//  }
//
//  if (basePtr) {
//    void* blockEnd = basePtr + basePtrInfo->count * basePtrInfo->typeSize;
//    // Ensure that the given address is in bounds and points to the start of an element
//    if (ptr < blockEnd && ((long)ptr - (long)basePtr) % basePtrInfo->typeSize == 0) {
//      return basePtrInfo;
//    }
//  }
//
//  return nullptr;
//}

lookup_result MustSupportRT::getTypeInfoInternal(const void *baseAddr, int offset, const StructTypeInfo &containerInfo,
                                                 must::TypeInfo *type) const
{
  std::cerr << "Type is " << containerInfo.name << std::endl;

  // Find contained sub type
  int i = 0;
  int baseOffset = 0;
  for (; i < containerInfo.numMembers; i++){
    baseOffset = containerInfo.offsets[i];
    if (baseOffset >= offset) {
      break;
    }
  }



  assert(baseOffset >= offset && "Something went wrong with the base address computation");


  if (baseOffset > offset) {
    // Offset corresponds to location inside the member
    i--;
    baseOffset = containerInfo.offsets[i]; // Get last matching offset

    std::cerr << "Offset=" << offset << ", detected base offset=" << baseOffset << std::endl;

    auto memberInfo = containerInfo.memberTypes[i];
    if (memberInfo.kind == STRUCT) {
      auto memberStructInfo = typeConfig.getStructInfo(memberInfo.id);
      return getTypeInfoInternal(baseAddr + baseOffset, offset - baseOffset, memberStructInfo, type);
    } else {

      int dif = offset - baseOffset;
      int typeSize = typeConfig.getBuiltinTypeSize(memberInfo.id);
      // TODO: Pointers
      if (dif % typeSize == 0) {
        *type = memberInfo;
        return SUCCESS;
      }
      std::cerr << "Internal alignment wrong: type size is " << typeSize << std::endl;
      return BAD_ALIGNMENT;
    }
  }

  // Offset corresponds directly to a member
  *type = containerInfo.memberTypes[i];
  return SUCCESS;
}

lookup_result MustSupportRT::getTypeInfo(const void* addr, must::TypeInfo* type, int* count) const {

  auto it = typeMap.find(addr);
  if (it != typeMap.end()) {
    auto ptrInfo = &it->second;

    auto typeInfo = typeConfig.getTypeInfo(ptrInfo->typeId);

    *type = typeInfo;
    *count = ptrInfo->count;
    return SUCCESS;

  }

    // The given pointer does not correspond to the start of an array -> find possible base pointer


  // TODO: More efficient lookup?
  const void* basePtr = 0;
  const PointerInfo* basePtrInfo = nullptr;
  for (auto it = typeMap.begin(); it != typeMap.end(); it++) {
    if (it->first < addr && it->first > basePtr) {
      basePtr = it->first;
      basePtrInfo = &it->second;
    }
  }

  if (basePtr) {
    const void* blockEnd = basePtr + basePtrInfo->count * basePtrInfo->typeSize;
    // Ensure that the given address is in bounds and points to the start of an element
    if (addr >= blockEnd) {
      return UNKNOWN_ADDRESS;
    }
    long addrDif = (long)addr - (long)basePtr;
    int internalOffset = addrDif % basePtrInfo->typeSize; // Offset of the pointer w.r.t. the start of the containing type
    if (internalOffset != 0) {  // TODO: Ensure that this works correctly with alignment
      if (typeConfig.isBuiltinType(basePtrInfo->typeId)) {
        std::cerr << "Primitive alignment wrong" << std::endl;
        return BAD_ALIGNMENT; // Address points to the middle of a builtin type
      } else  {
        const void* structAddr = basePtr + addrDif - internalOffset;
        auto structInfo = typeConfig.getStructInfo(basePtrInfo->typeId);

        auto result = getTypeInfoInternal(structAddr, internalOffset, structInfo, type);
        *count = 1;
        return result;
        // TODO: How to handle count?

      }
      // TODO: pointers
    } else {
      // Compute the element count from the given address
      *count = basePtrInfo->count - addrDif / basePtrInfo->typeSize;
      *type = typeConfig.getTypeInfo(basePtrInfo->typeId);
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

lookup_result MustSupportRT::resolveType(int id, int* len, must::TypeInfo* types[], int* count[], int* offsets[],
                                         int* extent) {
  TypeInfo typeInfo = typeConfig.getTypeInfo(id);
  // Requested ID must correspond to a struct
  if (typeInfo.kind != STRUCT) {
    return WRONG_KIND;
  }

  StructTypeInfo structInfo = typeConfig.getStructInfo(id);
  // TODO: Error handling
  //*len = structInfo.arraySizes;
  //*types =
  // TODO: Figure out how to store struct info efficiently
  return SUCCESS;
}

std::string MustSupportRT::getTypeName(int id) const {
  return typeConfig.getTypeName(id);
}

// bool MustSupportRT::checkType(void* ptr, int typeId) const {
//  auto info = getPtrInfo(ptr);
//  if (info) {
//    return info->typeId == typeId;
//  }
//  return false;
//}
//
// bool MustSupportRT::checkType(void* ptr, std::string typeName) const {
//  // int id = typeConfig.getTypeID(typeName);
//  // return checkType(ptr, id);
//  // TODO
//  return false;
//}

void MustSupportRT::onAlloc(void* addr, int typeId, long count, long typeSize) {
  // TODO: Check if entry already exists
  typeMap[addr] = {addr, typeId, count, typeSize};
  auto typeString = typeConfig.getTypeName(typeId);
  std::cout << "Alloc    " << addr << "   " << typeString << "   " << typeSize << "     " << count << std::endl;
}

void MustSupportRT::onFree(void* addr) {
  auto it = typeMap.find(addr);
  if (it != typeMap.end()) {
    typeMap.erase(it);
    std::cout << "Free     " << addr << std::endl;
  }
  // TODO: What to do when not found?
}
}
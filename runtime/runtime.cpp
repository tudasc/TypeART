#include "runtime.h"

#include <iostream>

void __must_support_alloc(void* addr, int typeId, long count, long typeSize) {
  must::MustSupportRT::get().onAlloc(addr, typeId, count, typeSize);
}

void __must_support_dealloc(void* addr) {
  must::MustSupportRT::get().onDealloc(addr);
}

int mustCheckType(void* addr, int typeId) {
  return must::MustSupportRT::get().checkType(addr, typeId);
}

namespace must {

MustSupportRT::MustSupportRT() {
  std::cout << "MUST Support Runtime Trace" << std::endl;
  std::cout << "**************************" << std::endl;
  std::cout << "Operation       Address       Type       Size      Count" << std::endl;
  std::cout << "--------------------------------------------------------" << std::endl;
}

const TypeInfo* MustSupportRT::getTypeInfo(void* ptr) const {
  auto it = typeMap.find(ptr);
  if (it != typeMap.end()) {
    return &it->second;
  }

  // TODO: More efficient lookup?
  // Find possible base pointer
  const TypeInfo* basePtrInfo = nullptr;
  void* basePtr = 0;
  for (it = typeMap.begin(); it != typeMap.end(); it++) {
    if (it->first < ptr && it->first > basePtr) {
      basePtr = it->first;
      basePtrInfo = &it->second;
    }
  }

  if (basePtr) {
    void* blockEnd = basePtr + basePtrInfo->count * basePtrInfo->typeSize;
    // Ensure that the given address is in bounds and points to the start of an element
    if (ptr < blockEnd && ((long)ptr - (long)basePtr) % basePtrInfo->typeSize == 0) {
      return basePtrInfo;
    }
  }

  return nullptr;
}

bool MustSupportRT::checkType(void* ptr, int typeId) const {
  auto info = getTypeInfo(ptr);
  if (info) {
    return info->typeId == typeId;
  }
  return false;
}

void MustSupportRT::onAlloc(void* addr, int typeId, long count, long typeSize) {
  // TODO: Check if entry already exists
  typeMap[addr] = {addr, typeId, count, typeSize};
  std::cout << "Allocation\t" << addr << "\t" << typeId << "\t" << typeSize << "\t" << count << std::endl;
}

void MustSupportRT::onDealloc(void* addr) {
  auto it = typeMap.find(addr);
  if (it != typeMap.end()) {
    typeMap.erase(it);
  }
  // TODO: What to do when not found?
}
}
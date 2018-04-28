#include "runtime.h"
#include "../configio/ConfigIO.h"

#include <iostream>

void __must_support_alloc(void* addr, int typeId, long count, long typeSize) {
  must::MustSupportRT::get().onAlloc(addr, typeId, count, typeSize);
}

void __must_support_free(void* addr) {
  must::MustSupportRT::get().onFree(addr);
}

int mustCheckType(void* addr, int typeId) {
  return must::MustSupportRT::get().checkType(addr, typeId);
}

int mustCheckTypeName(void* addr, const char* typeName) {
  return must::MustSupportRT::get().checkType(addr, typeName);
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

const PointerInfo* MustSupportRT::getPtrInfo(void *ptr) const {
  auto it = typeMap.find(ptr);
  if (it != typeMap.end()) {
    return &it->second;
  }

  // TODO: More efficient lookup?
  // Find possible base pointer
  const PointerInfo* basePtrInfo = nullptr;
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
  auto info = getPtrInfo(ptr);
  if (info) {
    return info->typeId == typeId;
  }
  return false;
}

bool MustSupportRT::checkType(void* ptr, std::string typeName) const {
  // int id = typeConfig.getTypeID(typeName);
  // return checkType(ptr, id);
  // TODO
  return false;
}

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
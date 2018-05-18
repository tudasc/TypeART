#ifndef RUNTIME_RUNTIME_H_
#define RUNTIME_RUNTIME_H_

#include "RuntimeInterface.h"
#include <TypeConfig.h>
#include <map>

extern "C" {
void __must_support_alloc(void* addr, int typeId, long count, long typeSize);
void __must_support_free(void* addr);
}

namespace must {

struct PointerInfo {
  void* addr;
  int typeId;
  long count;
  long typeSize;
};

class MustSupportRT {
 public:
  using LookupResult = lookup_result;

  static MustSupportRT& get() {
    static MustSupportRT instance;
    return instance;
  }

  // bool checkType(void* ptr, int typeID) const;
  // bool checkType(void* ptr, std::string typeName) const;

  // const PointerInfo* getPtrInfo(void *ptr) const;

  LookupResult getTypeInfo(const void* addr, must::TypeInfo* type, int* count) const;

  LookupResult getBuiltinInfo(const void* addr, must::BuiltinType* type) const;
  LookupResult getStructInfo(int id, const StructTypeInfo** structInfo) const;
  // LookupResult resolveType(int id, int* len, must::TypeInfo* types[], int* count[], int* offsets[], int* extent);
  std::string getTypeName(int id) const;

  void onAlloc(void* addr, int typeID, long count, long typeSize);
  void onFree(void* addr);

 private:
  MustSupportRT();

  LookupResult getTypeInfoInternal(const void* baseAddr, int offset, const StructTypeInfo& containingType,
                                   must::TypeInfo* type) const;

  void printTraceStart();

  TypeConfig typeConfig;
  std::map<const void*, PointerInfo> typeMap;
};
}

#endif
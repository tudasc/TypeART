#ifndef RUNTIME_RUNTIME_H_
#define RUNTIME_RUNTIME_H_

#include "../configio/TypeConfig.h"
#include "RuntimeInterface.h"
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
  LookupResult resolveType(int id, int* len, must::TypeInfo* types[], int* count[], size_t* offsets[], size_t* extent);
  std::string getTypeName(int id) const;

  void onAlloc(void* addr, int typeID, long count, long typeSize);
  void onFree(void* addr);

 private:
  MustSupportRT();

  void printTraceStart();

  TypeConfig typeConfig;
  std::map<const void*, PointerInfo> typeMap;
};
}

#endif
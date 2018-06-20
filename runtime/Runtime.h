#ifndef RUNTIME_RUNTIME_H_
#define RUNTIME_RUNTIME_H_

#include "RuntimeInterface.h"
#include <TypeDB.h>

#include <map>


extern "C" {
void __must_support_alloc(void* addr, int typeId, size_t count, size_t typeSize);
void __must_support_free(void* addr);
}

namespace must {

struct PointerInfo {
  const void* addr;
  int typeId;
  size_t count;
  size_t typeSize;
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

  LookupResult getTypeInfo(const void* addr, must::TypeInfo* type, size_t* count) const;

  LookupResult getBuiltinInfo(const void* addr, must::BuiltinType* type) const;

  LookupResult getStructInfo(int id, const StructTypeInfo** structInfo) const;

  // LookupResult resolveType(int id, int* len, must::TypeInfo* types[], int* count[], int* offsets[], int* extent);

  const std::string& getTypeName(int id) const;

  void onAlloc(void* addr, int typeID, size_t count, size_t typeSize);

  void onFree(void* addr);

 private:
  MustSupportRT();

  LookupResult getTypeInfoInternal(const void* baseAddr, size_t offset, const StructTypeInfo& containingType,
                                   must::TypeInfo* type, size_t* count) const;

  void printTraceStart();

  bool loadTypes(const std::string& file);

  const void* findBaseAddress(const void* addr) const;

  TypeDB typeDB;
  std::map<const void*, PointerInfo> typeMap;

  std::string typeFileName{"musttypes"};
};

}  // namespace must

#endif

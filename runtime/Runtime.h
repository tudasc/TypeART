#ifndef RUNTIME_RUNTIME_H_
#define RUNTIME_RUNTIME_H_

#include "RuntimeInterface.h"
#include <TypeDB.h>

#include <deque>
#include <map>

extern "C" {
void __typeart_alloc(void* addr, int typeId, size_t count, size_t typeSize, int isLocal);
void __typeart_free(void* addr);
void __typeart_enter_scope();
void __typeart_leave_scope();
}

namespace typeart {

struct PointerInfo {
  const void* addr;
  int typeId;
  size_t count;
  size_t typeSize;
};

class TypeArtRT {
 public:
  using LookupResult = lookup_result;

  static TypeArtRT& get() {
    static TypeArtRT instance;
    return instance;
  }

  // bool checkType(void* ptr, int typeID) const;
  // bool checkType(void* ptr, std::string typeName) const;

  // const PointerInfo* getPtrInfo(void *ptr) const;

  LookupResult getTypeInfo(const void* addr, typeart::TypeInfo* type, size_t* count) const;

  LookupResult getBuiltinInfo(const void* addr, typeart::BuiltinType* type) const;

  LookupResult getStructInfo(int id, const StructTypeInfo** structInfo) const;

  // LookupResult resolveType(int id, int* len, typeart::TypeInfo* types[], int* count[], int* offsets[], int* extent);

  const std::string& getTypeName(int id) const;

  void onAlloc(const void* addr, int typeID, size_t count, size_t typeSize, bool isLocal);

  void onFree(const void* addr);

  void onEnterScope();

  void onLeaveScope();

 private:
  TypeArtRT();

  LookupResult getTypeInfoInternal(const void* baseAddr, size_t offset, const StructTypeInfo& containingType,
                                   typeart::TypeInfo* type, size_t* count) const;

  void printTraceStart();

  bool loadTypes(const std::string& file);

  const void* findBaseAddress(const void* addr) const;

  TypeDB typeDB;

  std::map<const void*, PointerInfo> typeMap;

  std::deque<std::vector<const void*>> scopes;

  std::string typeFileName{"types.yaml"};
};

}  // namespace typeart

#endif

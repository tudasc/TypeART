#ifndef RUNTIME_RUNTIME_H_
#define RUNTIME_RUNTIME_H_

#include "../configio/TypeConfig.h"
#include <map>

extern "C" {
void __must_support_alloc(void* addr, int typeId, long count, long typeSize);
void __must_support_free(void* addr);

// C interface
int mustCheckType(void* addr, int typeId);
int mustCheckTypeName(void* addr, const char* typeName);

void must_support_get_builtin_type(const void* addr, must::BuiltinType* type);
void must_support_get_type(const void* addr, must::TypeInfo* type, int* count);
void must_support_resolve_type(int id, int* len, must::TypeInfo* types[], int* count[], size_t* offsets[], size_t* extent);

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
  static MustSupportRT& get() {
    static MustSupportRT instance;
    return instance;
  }

  bool checkType(void* ptr, int typeID) const;
  bool checkType(void* ptr, std::string typeName) const;

  const PointerInfo* getPtrInfo(void *ptr) const;

  TypeInfo getTypeInfo(int typeID);


  void onAlloc(void* addr, int typeID, long count, long typeSize);
  void onFree(void* addr);

 private:
  MustSupportRT();

  void printTraceStart();

  TypeConfig typeConfig;
  std::map<void*, PointerInfo> typeMap;
};
}

#endif
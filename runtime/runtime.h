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
}

namespace must {

struct RTTypeInfo {
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

  bool checkType(void* ptr, int typeId) const;
  bool checkType(void* ptr, std::string typeName) const;

  const RTTypeInfo* getTypeInfo(void* ptr) const;

  void onAlloc(void* addr, int typeId, long count, long typeSize);
  void onFree(void* addr);

 private:
  MustSupportRT();

  void printTraceStart();

  TypeConfig typeConfig;
  std::map<void*, RTTypeInfo> typeMap;
};
}

#endif
#ifndef RUNTIME_RUNTIME_H_
#define RUNTIME_RUNTIME_H_

#include <map>

extern "C" {
void __must_support_alloc(void* addr, int typeId, long count, long typeSize);
void __must_support_free(void* addr);

// C interface
int mustCheckType(void* addr, int typeId);
}

namespace must {

struct TypeInfo {
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

  const TypeInfo* getTypeInfo(void* ptr) const;

  void onAlloc(void* addr, int typeId, long count, long typeSize);
  void onDealloc(void* addr);

 private:
  MustSupportRT();

  std::map<void*, TypeInfo> typeMap;
};
}

#endif
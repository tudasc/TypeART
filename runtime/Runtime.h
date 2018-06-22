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


    /**
     * Determines the type and array element count at the given address.
     * Depending on the result of the query, one of the following status codes is returned:
     *  - SUCCESS: The query was successful and the contents of type and count are valid.
     *  - UNKNOWN_ADDRESS: The given address is either not allocated, or was not correctly recorded by the runtime.
     *  - BAD_ALIGNMENT: The given address does not line up with the start of the atomic type at that location.
     */
  LookupResult getTypeInfo(const void* addr, typeart::TypeInfo* type, size_t* count) const;

    /**
     * Returns the builtin type at the given address. Returns WRONG_KIND, if the type is not a builtin.
     */
  LookupResult getBuiltinInfo(const void* addr, typeart::BuiltinType* type) const;

    /**
     * Returns information about the struct with the given ID.
     */
  LookupResult getStructInfo(int id, const StructTypeInfo** structInfo) const;

  // LookupResult resolveType(int id, int* len, typeart::TypeInfo* types[], int* count[], int* offsets[], int* extent);

  const std::string& getTypeName(int id) const;

  void onAlloc(const void* addr, int typeID, size_t count, size_t typeSize, bool isLocal);

  void onFree(const void* addr);

  void onEnterScope();

  void onLeaveScope();

 private:

  TypeArtRT();

    /**
     * If a given address points inside a known struct, this method is used to recursively resolve the exact member type.
     */
  LookupResult getTypeInfoInternal(const void* baseAddr, size_t offset, const StructTypeInfo& containingType,
                                   typeart::TypeInfo* type, size_t* count) const;

    /**
     * Finds the struct member corresponding to the given byte offset.
     * If the offset is greater than the extent of the struct, the last member is returned.
     * Therefore, the caller must either ensure that the given offset is valid or explicitly check for this case.
     */
    size_t getMemberIndex(const StructTypeInfo& structInfo, size_t offset) const;

  void printTraceStart();

    /**
     * Loads the type file created by the LLVM pass.
     */
  bool loadTypes(const std::string& file);

    /**
     * Given an address, this method searches for the pointer that corresponds to the start of the allocated block.
     * Returns null if the memory location is not registered as allocated.
     */
  const void* findBaseAddress(const void* addr) const;

  TypeDB typeDB;

  std::map<const void*, PointerInfo> typeMap;

  std::deque<std::vector<const void*>> scopes;

  std::string typeFileName{"types.yaml"};
};

}  // namespace typeart

#endif

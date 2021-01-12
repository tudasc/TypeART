#ifndef RUNTIME_RUNTIME_H_
#define RUNTIME_RUNTIME_H_

#include "CallbackInterface.h"
#include "RuntimeData.h"
#include "RuntimeInterface.h"
#include "typelib/TypeDB.h"

#include <vector>

namespace llvm {
template <typename T>
class Optional;
}  // namespace llvm

namespace typeart {

class TypeArtRT final {
  RuntimeT::PointerMap typeMap;
  RuntimeT::Stack stackVars;
  TypeDB typeDB;

  static constexpr const char* defaultTypeFileName = "types.yaml";

 public:
  enum class AllocState : unsigned {
    NO_INIT      = 1 << 0,
    OK           = 1 << 1,
    ADDR_SKIPPED = 1 << 2,
    NULL_PTR     = 1 << 3,
    ZERO_COUNT   = 1 << 4,
    NULL_ZERO    = 1 << 5,
    ADDR_REUSE   = 1 << 6,
    UNKNOWN_ID   = 1 << 7
  };

  enum class FreeState : unsigned {
    NO_INIT      = 1 << 0,
    OK           = 1 << 1,
    ADDR_SKIPPED = 1 << 2,
    NULL_PTR     = 1 << 3,
    UNREG_ADDR   = 1 << 4,
  };

  using TypeArtStatus = typeart_status;

  static TypeArtRT& get() {
    static TypeArtRT instance;
    return instance;
  }

  /**
   * Determines the type and array element count at the given address.
   * For nested types with classes/structs, the containing type is resolved recursively, until an exact with the address
   * is found.
   *
   * Note that this function will always return the outermost type lining up with the address.
   * Given a pointer to the start of a struct, the returned type will therefore be that of the struct, not of the first
   * member.
   *
   * Depending on the result of the query, one of the following status codes is returned:
   *  - TA_OK: The query was successful and the contents of type and count are valid.
   *  - TA_UNKNOWN_ADDRESS: The given address is either not allocated, or was not correctly recorded by the runtime.
   *  - TA_BAD_ALIGNMENT: The given address does not line up with the start of the atomic type at that location.
   *  - TA_INVALID_ID: Encountered unregistered ID during lookup.
   */
  TypeArtStatus getTypeInfo(const void* addr, int* type, size_t* count) const;

  /**
   * Determines the outermost type and array element count at the given address.
   * Unlike in getTypeInfo(), there is no further resolution of subtypes.
   * Instead, additional information about the position of the address within the containing type is returned.
   *
   * The starting address of the referenced array element can be deduced by computing `(size_t) addr - offset`.
   *
   * \param[in] addr The address.
   * \param[out] count Number of elements in the containing buffer, not counting elements before the given address.
   * \param[out] baseAddress Address of the containing buffer.
   * \param[out] offset The byte offset within that buffer element.
   *
   * \return A status code. For an explanation of errors, refer to getTypeInfo().
   *
   */
  TypeArtStatus getContainingTypeInfo(const void* addr, int* type, size_t* count, const void** baseAddress,
                                      size_t* offset) const;

  /**
   * Determines the subtype at the given offset w.r.t. a base address and a corresponding containing type.
   * Note that if the subtype is itself a struct, you may have to call this function again.
   * If it returns with *subTypeOffset == 0, the address has been fully resolved.
   *
   * \param[in] baseAddr Pointer to the start of the containing type.
   * \param[in] offset Byte offset within the containing type.
   * \param[in] containerInfo typeart_struct_layout corresponding to the containing type
   * \param[out] subType Type ID corresponding to the subtype.
   * \param[out] subTypeBaseAddr Pointer to the start of the subtype.
   * \param[out] subTypeOffset Byte offset within the subtype.
   * \param[out] subTypeCount Number of elements in subarray.
   *
   * \return One of the following status codes:
   *  - TA_OK: Success.
   *  - TA_BAD_ALIGNMENT: Address corresponds to location inside an atomic type or padding.
   *  - TA_BAD_OFFSET: The provided offset is invalid.
   */
  TypeArtStatus getSubTypeInfo(const void* baseAddr, size_t offset, typeart_struct_layout containerInfo, int* subType,
                               const void** subTypeBaseAddr, size_t* subTypeOffset, size_t* subTypeCount) const;

  /**
   * Wrapper function using StructTypeInfo.
   */
  TypeArtStatus getSubTypeInfo(const void* baseAddr, size_t offset, const StructTypeInfo& containerInfo, int* subType,
                               const void** subTypeBaseAddr, size_t* subTypeOffset, size_t* subTypeCount) const;

  /**
   * Returns the builtin type at the given address.
   *
   * \param[in] addr The address.
   * \param[out] type The builtin type.
   * \return TA_OK, if the type is a builtin, TA_WRONG_KIND otherwise.
   */
  TypeArtStatus getBuiltinInfo(const void* addr, typeart::BuiltinType* type) const;

  /**
   * Given a type ID, this function provides information about the corresponding struct type.
   *
   * \param[in] id The type ID.
   * \param[out] structInfo Pointer to the StructTypeInfo corresponding to the type ID.
   *
   * \return One of the following status codes:
   *  - TA_OK: Sucess.
   *  - TA_WRONG_KIND: ID does not correspond to a struct type.
   *  - TA_INVALID_ID: ID is not valid.
   */
  TypeArtStatus getStructInfo(int id, const StructTypeInfo** structInfo) const;

  /**
   * Returns the stored debug address generated by __builtin_return_address(0).
   *
   * \param[in] addr The address.
   * \param[out] retAddr The approximate address where the allocation occurred, or nullptr.
   */
  void getReturnAddress(const void* addr, const void** retAddr) const;

  // TypeArtStatus resolveType(int id, int* len, typeart::TypeInfo* types[], int* count[], int* offsets[], int* extent);

  /**
   * Returns the name of the type corresponding to the given ID.
   * This can be used for debugging and error messages.
   *
   * \param[in] id The type ID.
   * \return The name of the type.
   */
  const std::string& getTypeName(int id) const;

  size_t getTypeSize(int id) const;

  void onAlloc(const void* addr, int typeID, size_t count, const void* retAddr);

  void onAllocStack(const void* addr, int typeID, size_t count, const void* retAddr);

  void onAllocGlobal(const void* addr, int typeID, size_t count, const void* retAddr);

  void onFreeHeap(const void* addr, const void* retAddr);

  void onLeaveScope(int alloca_count, const void* retAddr);

 private:
  TypeArtRT();
  ~TypeArtRT();

  /**
   * If a given address points inside a known struct, this method is used to recursively resolve the exact member type.
   */
  TypeArtStatus getTypeInfoInternal(const void* baseAddr, size_t offset, const StructTypeInfo& containingType,
                                    int* type, size_t* count) const;

  /**
   * Finds the struct member corresponding to the given byte offset.
   * If the offset is greater than the extent of the struct, the last member is returned.
   * Therefore, the caller must either ensure that the given offset is valid or explicitly check for this case.
   */
  size_t getMemberIndex(typeart_struct_layout structInfo, size_t offset) const;

  /**
   * Loads the type file created by the LLVM pass.
   */
  bool loadTypes(const std::string& file);

  inline AllocState doAlloc(const void* addr, int typeID, size_t count, const void* retAddr);

  template <bool stack>
  inline FreeState doFree(const void* addr, const void* retAddr);

  /**
   * Given an address, this method searches for the pointer that corresponds to the start of the allocated block.
   * Returns null if the memory location is not registered as allocated.
   */
  llvm::Optional<RuntimeT::MapEntry> findBaseAddress(const void* addr) const;
};

}  // namespace typeart

#endif

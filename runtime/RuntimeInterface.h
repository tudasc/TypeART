#ifndef LLVM_MUST_SUPPORT_RUNTIMEINTERFACE_H
#define LLVM_MUST_SUPPORT_RUNTIMEINTERFACE_H

#include "../typelib/TypeInterface.h"

#ifdef __cplusplus
#include <cstddef>
#else
#include <stddef.h>
#endif

// Type assert macro
#ifndef NDEBUG
// TODO: Unique name needed for type ptr?
#define ASSERT_TYPE(ptr, type)                   \
  {                                              \
    type* __type_ptr;                            \
    __typeart_assert_type_stub(ptr, __type_ptr); \
  }

#define ASSERT_TYPE(ptr, type, len)                   \
  {                                              \
    type* __type_ptr;                            \
    __typeart_assert_type_stub_len(ptr, __type_ptr, len); \
  }
#else
#define ASSERT_TYPE(ptr, type)
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum typeart_status_t {  // NOLINT
  TA_OK,
  TA_UNKNOWN_ADDRESS,
  TA_BAD_ALIGNMENT,
  TA_BAD_OFFSET,
  TA_WRONG_KIND,
  TA_INVALID_ID
} typeart_status;

typedef struct typeart_struct_layout_t {  // NOLINT
  int id;
  const char* name;
  size_t extent;
  size_t len;
  const size_t* offsets;
  const int* member_types;
  const size_t* count;
} typeart_struct_layout;

/**
 * \deprecated{Use typeart_get_type instead and check the returned type}
 *
 * Returns the builtin type at the given address.
 *
 * \param[in] addr The address.
 * \param[out] type The builtin type.
 * \return TA_OK, if the type is a builtin, TA_WRONG_KIND otherwise.
 */
typeart_status typeart_get_builtin_type(const void* addr, typeart_builtin_type* type);

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
typeart_status typeart_get_type(const void* addr, int* type, size_t* count);

/**
 * Determines the outermost type and array element count at the given address.
 * Unlike in typeart_get_type(), there is no further resolution of subtypes.
 * Instead, additional information about the position of the address within the containing type is returned.
 *
 * The starting address of the referenced array element can be deduced by computing `(size_t) addr - offset`.
 *
 * \param[in] addr The address.
 * \param[out] count Number of elements in the containing buffer, not counting elements before the given address.
 * \param[out] base_address Address of the containing buffer.
 * \param[out] offset The byte offset within that buffer element.
 *
 * \return A status code. For an explanation of errors, refer to typeart_get_type().
 *
 */
typeart_status typeart_get_containing_type(const void* addr, int* type, size_t* count, const void** base_address,
                                           size_t* offset);

/**
 * Determines the subtype at the given offset w.r.t. a base address and a corresponding containing type.
 * Note that if the subtype is itself a struct, you may have to call this function again.
 * If it returns with *subTypeOffset == 0, the address has been fully resolved.
 *
 * \param[in] baseAddr Pointer to the start of the containing type.
 * \param[in] offset Byte offset within the containing type.
 * \param[in] containerLayout typeart_struct_layout corresponding to the containing type
 * \param[out] subtype The type ID corresponding to the subtype.
 * \param[out] subtype_base_addr Pointer to the start of the subtype.
 * \param[out] subtype_offset Byte offset within the subtype.
 * \param[out] subtype_count Number of elements in subarray.
 *
 * \return One of the following status codes:
 *  - TA_OK: Success.
 *  - TA_BAD_ALIGNMENT: Address corresponds to location inside an atomic type or padding.
 *  - TA_BAD_OFFSET: The provided offset is invalid.
 */
typeart_status typeart_get_subtype(const void* base_addr, size_t offset, typeart_struct_layout container_layout,
                                   int* subtype, const void** subtype_base_addr, size_t* subtype_offset,
                                   size_t* subtype_count);

/**
 * Given a type ID, this function provides information about the corresponding struct type.
 *
 * \param[in] id The type ID.
 * \param[out] struct_layout Data layout of the struct.
 *
 * \return One of the following status codes:
 *  - TA_OK: Sucess.
 *  - TA_WRONG_KIND: ID does not correspond to a struct type.
 *  - TA_INVALID_ID: ID is not valid.
 */
typeart_status typeart_resolve_type(int id, typeart_struct_layout* struct_layout);

/**
 * Returns the name of the type corresponding to the given ID.
 * This can be used for debugging and error messages.
 *
 * \param[in] id The type ID.
 * \return The name of the type.
 */
const char* typeart_get_type_name(int id);

/**
 * Returns the stored debug address generated by __builtin_return_address(0).
 *
 * \param[in] addr The address.
 * \param[out] retAddr The approximate address where the allocation occurred, or nullptr.
 */
void typeart_get_return_address(const void* addr, const void** retAddr);

void __typeart_assert_type_stub(const void* addr, const void* typePtr);
void __typeart_assert_type_stub_len(const void* addr, const void* typePtr, size_t);

#ifdef __cplusplus
}
#endif

#endif  // LLVM_MUST_SUPPORT_RUNTIMEINTERFACE_H

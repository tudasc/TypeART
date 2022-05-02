// TypeART library
//
// Copyright (c) 2017-2022 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#ifndef TYPEART_RUNTIMEINTERFACE_H
#define TYPEART_RUNTIMEINTERFACE_H

#include "TypeInterface.h"

#ifdef __cplusplus
#include <cstddef>
#else
#include <stdbool.h>
#include <stddef.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum typeart_status_t {  // NOLINT
  TYPEART_OK,
  TYPEART_UNKNOWN_ADDRESS,
  TYPEART_BAD_ALIGNMENT,
  TYPEART_BAD_OFFSET,
  TYPEART_WRONG_KIND,
  TYPEART_INVALID_ID,
  TYPEART_ERROR
} typeart_status;

typedef struct typeart_struct_layout_t {  // NOLINT
  int type_id;
  const char* name;
  size_t extent;
  size_t num_members;
  const size_t* offsets;
  const int* member_types;
  const size_t* count;
} typeart_struct_layout;

/**
 * Determines the type and array element count at the given address.
 * For nested types with classes/structs, the containing type is resolved recursively, until an exact with the address
 * is found.
 * See typeart_get_type_length and typeart_get_type_id for resolving only one such parameter
 *
 * Note that this function will always return the outermost type lining up with the address.
 * Given a pointer to the start of a struct, the returned type will therefore be that of the struct, not of the first
 * member.
 *
 * Code example:
 * {
 *   struct DataStruct { int a; double b; float c[2]; }; // sizeof(DataStruct) == 24 byte
 *   DataStruct data[5];
 *   We pass the address of the first element of the float array inside the struct:
 *   typeart_get_type(&data[1].c[0], &type_id, &count);
 *   returns:
 *   -> type_id: 5 (i.e., TYPEART_FLOAT)
 *   -> count: 2 (c[0] to end(c))
 * }
 *
 * \param[in] addr The address.
 * \param[out] type_id Type ID
 * \param[out] count Allocation size
 *
 * \return A status code:
 *  - TYPEART_OK: The query was successful and the contents of type and count are valid.
 *  - TYPEART_UNKNOWN_ADDRESS: The given address is either not allocated, or was not correctly recorded by the runtime.
 *  - TYPEART_BAD_ALIGNMENT: The given address does not line up with the start of the atomic type at that location.
 *  - TYPEART_INVALID_ID: Encountered unregistered ID during lookup.
 */
typeart_status typeart_get_type(const void* addr, int* type_id, size_t* count);

typeart_status typeart_get_type_length(const void* addr, size_t* count);

typeart_status typeart_get_type_id(const void* addr, int* type_id);

/**
 * Determines the outermost type and array element count at the given address.
 * Unlike in typeart_get_type(), there is no further resolution of subtypes.
 * Instead, additional information about the position of the address within the containing type is returned.
 *
 * The starting address of the referenced array element can be deduced by computing `(size_t) addr - offset`.
 * Note: The addr may point to an illegal memory location inside the containing type. This is not automatically
 * resolved, see code example below.
 *
 * Code example:
 * {
 *   Struct with implicit padding for correct alignment:
 *   struct DataStruct { int a; {4xbyte pad}; double b; float c[2]; }; // sizeof(DataStruct) == 24 byte
 *   DataStruct data[5];
 *   We pass the address of the first element of the float array inside the struct (containing type):
 *   typeart_get_containing_type(&data[1].c[0], &type_id, &count, base_address, &byte_offset);
 *   returns:
 *   -> type_id: 257 (or higher for struct types)
 *   -> count: 4 (including data[1] to data[4])
 *   -> base_addr: &data[0]
 *   -> byte_offset: 16 (counted from the containing type &data[1].c[0] to &data[1])
 *   Hence:
 *   ((addr - offset) - base_addr) == number of bytes to start of allocation "data" from data[1]
 *
 *   Example 2, illegal memory address is not explicitly caught by return state:
 *   typeart_get_containing_type(&data[0].a + sizeof(int), ...);
 *   returns:
 *   -> type_id: 257 (or higher for struct types)
 *   -> count: 5 (including data[0] to data[4])
 *   -> base_addr: &data[0]
 *   -> byte_offset: 4 (points to padding, counted from the containing type (&data[0].a + sizeof(int))
 *      to &data[0])
 * }
 *
 * \param[in] addr The address.
 * \param[out] type_id Type ID of the containing type
 * \param[out] count Number of elements in the containing buffer, not counting elements before the given address.
 * \param[out] base_address Address of the containing buffer.
 * \param[out] byte_offset The byte offset within that buffer element.
 *
 * \return A status code.
 *  - TYPEART_OK: The query was successful.
 *  - TYPEART_UNKNOWN_ADDRESS: The given address is either not allocated, or was not correctly recorded by the runtime.
 */
typeart_status typeart_get_containing_type(const void* addr, int* type_id, size_t* count, const void** base_address,
                                           size_t* byte_offset);

/**
 * Determines the subtype at the given offset w.r.t. a base address and a corresponding containing type.
 * Note that if the subtype is itself a struct, you may have to call this function again.
 * If it returns with *subTypeOffset == 0, the address has been fully resolved.
 *
 * Code example:
 * {
 *   struct DataStruct { int a; double b; float c[2]; }; // sizeof(DataStruct) == 24 byte
 *   DataStruct data[5];
 *   typeart_struct_layout layout_data;
 *   Determine layout of data:
 *   {
 *     int type_id;
 *     typeart_get_type_id(&data[0], &type_id);
 *     typeart_resolve_type_id(type_id, &layout_data);
 *   }
 *   We pass the address of the first element of the data array:
 *   status = typeart_get_subtype(&data[1], 20, &layout_data, &subtype_id, &subtype_base_addr, &subtype_byte_offset,
 *                                &subtype_count);
 *   returns:
 *   -> subtype_id: 5 (TYPEART_FLOAT)
 *   -> subtype_base_addr: &data[1].c[0]
 *   -> subtype_byte_offset: 0
 *   -> subtype_count: 1 (length of member float[2] at offset 20)
 * }
 *
 * \param[in] baseAddr Pointer to the start of the containing type.
 * \param[in] offset Byte offset within the containing type.
 * \param[in] container_layout typeart_struct_layout corresponding to the containing type
 * \param[out] subtype_id The type ID corresponding to the subtype.
 * \param[out] subtype_base_addr Pointer to the start of the subtype.
 * \param[out] subtype_byte_offset Byte offset within the subtype.
 * \param[out] subtype_count Number of elements in subarray.
 *
 * \return One of the following status codes:
 *  - TYPEART_OK: Success.
 *  - TYPEART_BAD_ALIGNMENT: Address corresponds to location inside an atomic type or padding.
 *  - TYPEART_BAD_OFFSET: The provided offset is invalid.
 *  - TYPEART_ERROR: The typeart_struct_layout is invalid.
 */
typeart_status typeart_get_subtype(const void* base_addr, size_t offset, const typeart_struct_layout* container_layout,
                                   int* subtype_id, const void** subtype_base_addr, size_t* subtype_byte_offset,
                                   size_t* subtype_count);

/**
 * Returns the stored debug address generated by __builtin_return_address(0).
 *
 * \param[in] addr The address.
 * \param[out] return_addr The approximate address where the allocation occurred, or nullptr.
 *
 * \return One of the following status codes:
 *  - TYPEART_OK: Success.
 *  - TYPEART_UNKNOWN_ADDRESS: The given address is either not allocated, or was not recorded by the runtime.
 */
typeart_status typeart_get_return_address(const void* addr, const void** return_addr);

/**
 * Tries to return file, function and line of a memory address from the current process.
 * Needs (1) either llvm-symbolizer or addr2line to be installed, and (2) target code should be compiled debug
 * information for useful output. Note: file, function, line are allocated with malloc. They need to be free'd by the
 * caller.
 *
 * \param[in] addr The address.
 * \param[out] file The file where the address was created at.
 * \param[out] function The function where the address was created at.
 * \param[out] line The approximate line where the address was created at.
 *
 * \return One of the following status codes:
 *  - TYPEART_OK: Success.
 *  - TYPEART_UNKNOWN_ADDRESS: The given address is either not allocated, or was not recorded by the runtime.
 *  - TYPEART_ERROR: Memory could not be allocated.
 */
typeart_status typeart_get_source_location(const void* addr, char** file, char** function, char** line);

/**
 * Given an address, this function provides information about the corresponding struct type.
 * This is more expensive than the below version, since the pointer addr must be resolved.
 *
 * \param[in] addr The pointer address
 * \param[out] struct_layout Data layout of the struct.
 *
 * \return One of the following status codes:
 *  - TYPEART_OK: Success.
 *  - TYPEART_WRONG_KIND: ID does not correspond to a struct type.
 *  - TYPEART_UNKNOWN_ADDRESS: The given address is either not allocated, or was not correctly recorded by the runtime.
 *  - TYPEART_BAD_ALIGNMENT: The given address does not line up with the start of the atomic type at that location.
 *  - TYPEART_INVALID_ID: Encountered unregistered ID during lookup.
 */
typeart_status typeart_resolve_type_addr(const void* addr, typeart_struct_layout* struct_layout);

/**
 * Given a type ID, this function provides information about the corresponding struct type.
 *
 * \param[in] type_id The type ID.
 * \param[out] struct_layout Data layout of the struct.
 *
 * \return One of the following status codes:
 *  - TYPEART_OK: Success.
 *  - TYPEART_WRONG_KIND: ID does not correspond to a struct type.
 *  - TYPEART_INVALID_ID: ID is not valid.
 */
typeart_status typeart_resolve_type_id(int type_id, typeart_struct_layout* struct_layout);

/**
 * Returns the name of the type corresponding to the given type ID.
 * This can be used for debugging and error messages.
 *
 * \param[in] type_id The type ID.
 * \return The name of the type, or "typeart_unknown_struct" if the ID is unknown.
 */
const char* typeart_get_type_name(int type_id);

/**
 * Returns true if this is a valid type according to
 * e.g., a built-in type or a user-defined type,
 * see also TypeInterface.h
 *
 * \param[in] type_id The type ID.
 * \return true, false
 */
bool typeart_is_valid_type(int type_id);

/**
 * Returns true if the type ID is in the pre-determined reserved range,
 * see TypeInterface.h
 *
 * \param[in] type_id The type ID.
 * \return true, false
 */
bool typeart_is_reserved_type(int type_id);

/**
 * Returns true if the type ID is a built-in type,
 * see TypeInterface.h
 *
 * \param[in] type_id The type ID.
 * \return true, false
 */
bool typeart_is_builtin_type(int type_id);

/**
 * Returns true if the type ID is a structure type.
 * Note: This can be a user-defined struct or class, or a
 * LLVM vector type. Use the below queries for specifics.
 *
 * \param[in] type_id The type ID.
 * \return true, false
 */
bool typeart_is_struct_type(int type_id);

/**
 * Returns true if the type ID is a user-defined structure type
 * (struct, class etc.)
 *
 * \param[in] type_id The type ID.
 * \return true, false
 */
bool typeart_is_userdefined_type(int type_id);

/**
 * Returns true if the type ID is a LLVM SIMD vector type
 *
 * \param[in] type_id The type ID.
 * \return true, false
 */
bool typeart_is_vector_type(int type_id);

/**
 * Returns the byte size of the type behind the ID.
 *
 * \param[in] type_id The type ID.
 * \return size in bytes of the type
 */
size_t typeart_get_type_size(int type_id);

/**
 * Version string "major.minor(.patch)" of TypeART.
 *
 * \return version string
 */
const char* typeart_get_project_version();

/**
 * Short Git revision (length: 10) string of TypeART.
 * Char "+" is appended, if uncommitted changes were detected.
 * Returns "N/A" if revision couldn't be generated.
 *
 * \return revision string
 */
const char* typeart_get_git_revision();

/**
 * Version string "major.minor" of LLVM used to build TypeART.
 *
 * \return version string
 */
const char* typeart_get_llvm_version();

#ifdef __cplusplus
}
#endif

#endif  // TYPEART_RUNTIMEINTERFACE_H

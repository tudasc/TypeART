#ifndef TYPEART_TYPEINTERFACE_H
#define TYPEART_TYPEINTERFACE_H

#ifdef __cplusplus
#include <cstddef>
#else
#include <stdbool.h>
#include <stddef.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum typeart_builtin_type_t {  // NOLINT
  TA_INT8  = 0,
  TA_INT16 = 1,
  TA_INT32 = 2,
  TA_INT64 = 3,

  // Note: Unsigned types are currently not supported
  // TA_UINT8,
  // TA_UINT16,
  // TA_UINT32,
  // TA_UINT64,

  TA_HALF      = 4,  // IEEE 754 half precision floating point type
  TA_FLOAT     = 5,  // IEEE 754 single precision floating point type
  TA_DOUBLE    = 6,  // IEEE 754 double precision floating point type
  TA_FP128     = 7,  // IEEE 754 quadruple precision floating point type
  TA_X86_FP80  = 8,  // x86 extended precision 80-bit floating point type
  TA_PPC_FP128 = 9,  // ICM extended precision 128-bit floating point type

  TA_PTR = 10,  // Represents all pointer types

  TA_NUM_VALID_IDS = TA_PTR + 1,

  TA_UNKNOWN_TYPE     = 255,
  TA_NUM_RESERVED_IDS = TA_UNKNOWN_TYPE + 1
} typeart_builtin_type;

/**
 * Returns the name of the type corresponding to the given type ID.
 * This can be used for debugging and error messages.
 *
 * \param[in] id The type ID.
 * \return The name of the type, or "typeart_unknown_struct" if the ID is unknown.
 */
const char* typeart_get_type_name(int id);

/**
 * Returns true if this is a valid type according to
 * e.g., a built-in type or a user-defined type,
 * see also TypeInterface.h
 *
 * \param[in] id The type ID.
 * \return true, false
 */
bool typeart_is_valid_type(int id);

/**
 * Returns true if the type ID is in the pre-determined reserved range,
 * see TypeInterface.h
 *
 * \param[in] id The type ID.
 * \return true, false
 */
bool typeart_is_reserved_type(int id);

/**
 * Returns true if the type ID is a built-in type,
 * see TypeInterface.h
 *
 * \param[in] id The type ID.
 * \return true, false
 */
bool typeart_is_builtin_type(int id);

/**
 * Returns true if the type ID is a structure type.
 * Note: This can be a user-defined struct or class, or a
 * LLVM vector type. Use the below queries for specifics.
 *
 * \param[in] id The type ID.
 * \return true, false
 */
bool typeart_is_struct_type(int id);

/**
 * Returns true if the type ID is a user-defined structure type
 * (struct, class etc.)
 *
 * \param[in] id The type ID.
 * \return true, false
 */
bool typeart_is_userdefined_type(int id);

/**
 * Returns true if the type ID is a LLVM SIMD vector type
 *
 * \param[in] id The type ID.
 * \return true, false
 */
bool typeart_is_vector_type(int id);

/**
 * Returns the byte size of the type behind the ID.
 *
 * \param[in] id The type ID.
 * \return size in bytes of the type
 */
size_t typeart_get_type_size(int id);

#ifdef __cplusplus
}
#endif

#endif  // TYPEART_TYPEINTERFACE_H

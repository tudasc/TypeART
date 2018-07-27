#ifndef LLVM_MUST_SUPPORT_TYPEINTERFACE_H
#define LLVM_MUST_SUPPORT_TYPEINTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

// extern const int TA_NUM_RESERVED_IDS;

// Type UNKNOWN is used for pointer types, when the underlying type is not specified.
// In conjunction with kind BUILTIN, UNKNOWN signifies an invalid type.
typedef enum typeart_builtin_type_t {  // NOLINT
  TA_INT8 = 0,
  TA_INT16 = 1,
  TA_INT32 = 2,
  TA_INT64 = 3,

  // Note: Unsigned types are currently not supported
  // TA_UINT8,
  // TA_UINT16,
  // TA_UINT32,
  // TA_UINT64,

  TA_HALF = 4,       // IEEE 754 half precision floating point type
  TA_FLOAT = 5,      // IEEE 754 single precision floating point type
  TA_DOUBLE = 6,     // IEEE 754 double precision floating point type
  TA_FP128 = 7,      // IEEE 754 quadruple precision floating point type
  TA_X86_FP80 = 8,   // x86 extended precision 80-bit floating point type
  TA_PPC_FP128 = 9,  // ICM extended precision 128-bit floating point type

  TA_PTR = 10,  // Represents all pointer types

  TA_NUM_VALID_IDS = TA_PTR + 1,

  TA_UNKNOWN_TYPE = 255,
  TA_NUM_RESERVED_IDS = TA_UNKNOWN_TYPE + 1
} typeart_builtin_type;

// typedef enum typeart_type_kind_t { BUILTIN, STRUCT, POINTER } typeart_type_kind;  // NOLINT

// typedef struct typeart_type_info_t {  // NOLINT
//  typeart_type_kind kind;
//  int id;
//} typeart_type_info;

#ifdef __cplusplus
}
#endif

#endif  // LLVM_MUST_SUPPORT_TYPEINTERFACE_H

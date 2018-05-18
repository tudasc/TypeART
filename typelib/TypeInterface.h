#ifndef LLVM_MUST_SUPPORT_TYPEINTERFACE_H
#define LLVM_MUST_SUPPORT_TYPEINTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

// TODO: Support for more types (e.g. long double)
typedef enum must_builtin_type_t {
  C_CHAR = 0,
  C_UCHAR = 1,
  C_SHORT = 2,
  C_USHORT = 3,
  C_INT = 4,
  C_UINT = 5,
  C_LONG = 6,
  C_ULONG = 7,
  C_FLOAT = 8,
  C_DOUBLE = 9,
  INVALID = 10,
  N_BUILTIN_TYPES
} must_builtin_type;

typedef enum must_type_kind_t { BUILTIN, STRUCT, POINTER } must_type_kind;

typedef struct must_type_info_t {
  must_type_kind kind;
  int id;
} must_type_info;

#ifdef __cplusplus
}
#endif

#endif  // LLVM_MUST_SUPPORT_TYPEINTERFACE_H

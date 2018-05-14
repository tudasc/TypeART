#ifndef LLVM_MUST_SUPPORT_TYPEINTERFACE_H
#define LLVM_MUST_SUPPORT_TYPEINTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum must_builtin_type_t {
  C_INT = 0,
  C_UINT = 1,
  C_CHAR = 2,
  C_UCHAR = 3,
  C_LONG = 4,
  C_ULONG = 5,
  C_FLOAT = 6,
  C_DOUBLE = 7,
  INVALID = 8,
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

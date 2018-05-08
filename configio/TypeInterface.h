#ifndef LLVM_MUST_SUPPORT_TYPEINTERFACE_H
#define LLVM_MUST_SUPPORT_TYPEINTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum must_builtin_type_t {
    C_INT, C_UINT, C_CHAR, C_UCHAR, C_LONG, C_ULONG, C_FLOAT, C_DOUBLE, INVALID, N_BUILTIN_TYPES
} must_builtin_type;

typedef enum must_type_kind_t {
    BUILTIN, STRUCT, POINTER
} must_type_kind;


typedef struct must_type_info_t {
    must_type_kind kind;
    int id;
} must_type_info;

#ifdef __cplusplus
}
#endif

#endif //LLVM_MUST_SUPPORT_TYPEINTERFACE_H

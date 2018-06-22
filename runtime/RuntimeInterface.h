#ifndef LLVM_MUST_SUPPORT_RUNTIMEINTERFACE_H
#define LLVM_MUST_SUPPORT_RUNTIMEINTERFACE_H

#include "TypeInterface.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum typeart_status_t { SUCCESS, UNKNOWN_ADDRESS, BAD_ALIGNMENT, WRONG_KIND, INVALID_ID } typeart_status;

typeart_status typeart_get_builtin_type(const void* addr, typeart_builtin_type* type);
typeart_status typeart_get_type(const void* addr, typeart_type_info* type, size_t* count);
typeart_status typeart_resolve_type(int id, size_t* len, const typeart_type_info** types, const size_t** count,
                                    const size_t** offsets, size_t* extent);
// lookup_result typeart_support_resolve_type_alloc_buffer(int id, int* len, typeart_type_info* types[], int* count[],
// int* offsets[],
//                                                     int* extent);

const char* typeart_get_type_name(int id);

#ifdef __cplusplus
}
#endif

#endif  // LLVM_MUST_SUPPORT_RUNTIMEINTERFACE_H

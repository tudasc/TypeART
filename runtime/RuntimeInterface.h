#ifndef LLVM_MUST_SUPPORT_RUNTIMEINTERFACE_H
#define LLVM_MUST_SUPPORT_RUNTIMEINTERFACE_H

#include "../typelib/TypeInterface.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum lookup_result_t { SUCCESS, UNKNOWN_ADDRESS, BAD_ALIGNMENT, WRONG_KIND, BUFFER_TO_SMALL } lookup_result;

lookup_result must_support_get_builtin_type(const void* addr, must_builtin_type* type);
lookup_result must_support_get_type(const void* addr, must_type_info* type, int* count);
lookup_result must_support_resolve_type(int id, int* len, const must_type_info* types[], const int* count[],
                                        const int* offsets[], int* extent);
// lookup_result must_support_resolve_type_alloc_buffer(int id, int* len, must_type_info* types[], int* count[], int*
// offsets[],
//                                                     int* extent);

const char* must_support_get_type_name(int id);

#ifdef __cplusplus
}
#endif

#endif  // LLVM_MUST_SUPPORT_RUNTIMEINTERFACE_H

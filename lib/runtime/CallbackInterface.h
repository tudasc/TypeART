//
// Created by ahueck on 16.10.20.
//

#ifndef TYPEART_CALLBACKINTERFACE_H
#define TYPEART_CALLBACKINTERFACE_H

#ifdef __cplusplus
#include <cstddef>
#else
#include <stddef.h>
#endif

// Callback function signatures invoked by the LLVM pass
#ifdef __cplusplus
extern "C" {
#endif
void __typeart_alloc(const void* addr, int type_id, size_t count);

void __typeart_alloc_global(const void* addr, int type_id, size_t count);
void __typeart_free(const void* addr);

void __typeart_alloc_stack(const void* addr, int type_id, size_t count);
void __typeart_leave_scope(int alloca_count);

// Called from OpenMP context
void __typeart_alloc_omp(const void* addr, int type_id, size_t count);
void __typeart_free_omp(const void* addr);
void __typeart_alloc_stack_omp(const void* addr, int type_id, size_t count);
void __typeart_leave_scope_omp(int alloca_count);
#ifdef __cplusplus
}
#endif

#endif  // TYPEART_CALLBACKINTERFACE_H

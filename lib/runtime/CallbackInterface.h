// TypeART library
//
// Copyright (c) 2017-2025 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#ifndef TYPEART_CALLBACKINTERFACE_H
#define TYPEART_CALLBACKINTERFACE_H

#include "RuntimeExport.h"

#ifdef __cplusplus
#include <cstddef>
#else
#include <stddef.h>
#endif

// Callback function signatures invoked by the LLVM pass
#ifdef __cplusplus
extern "C" {
#endif
TYPEART_EXPORT void __typeart_alloc(const void* addr, int type_id, size_t count);

TYPEART_EXPORT void __typeart_alloc_global(const void* addr, int type_id, size_t count);
TYPEART_EXPORT void __typeart_free(const void* addr);

TYPEART_EXPORT void __typeart_alloc_stack(const void* addr, int type_id, size_t count);
TYPEART_EXPORT void __typeart_leave_scope(int alloca_count);

// Called from OpenMP context
TYPEART_EXPORT void __typeart_alloc_omp(const void* addr, int type_id, size_t count);
TYPEART_EXPORT void __typeart_free_omp(const void* addr);
TYPEART_EXPORT void __typeart_alloc_stack_omp(const void* addr, int type_id, size_t count);
TYPEART_EXPORT void __typeart_leave_scope_omp(int alloca_count);

// Called for inlined type definitions mode
TYPEART_EXPORT void __typeart_alloc_mty(const void* addr, const void* info, size_t count);
TYPEART_EXPORT void __typeart_alloc_global_mty(const void* addr, const void* info, size_t count);
TYPEART_EXPORT void __typeart_alloc_stack_mty(const void* addr, const void* info, size_t count);
TYPEART_EXPORT void __typeart_register_type(const void* type);

TYPEART_EXPORT void __typeart_alloc_global_mty_omp(const void* addr, const void* info, size_t count);
TYPEART_EXPORT void __typeart_alloc_stack_mty_omp(const void* addr, const void* info, size_t count);
#ifdef __cplusplus
}
#endif

#endif  // TYPEART_CALLBACKINTERFACE_H

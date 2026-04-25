// TypeART library
//
// Copyright (c) 2017-2026 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#ifndef TYPEART_CUDARUNTIMEINTERFACE_H
#define TYPEART_CUDARUNTIMEINTERFACE_H

#include "RuntimeExport.h"
#include "RuntimeInterface.h"

#ifdef __cplusplus
#include <cstddef>
#else
#include <stdbool.h>
#include <stddef.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

TYPEART_EXPORT typeart_status typeart_cuda_is_device_ptr(const void* addr, bool* is_device_ptr_flag);

#ifdef __cplusplus
}
#endif

#endif  // TYPEART_CUDARUNTIMEINTERFACE_H

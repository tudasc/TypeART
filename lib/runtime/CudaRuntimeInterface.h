// TypeART library
//
// Copyright (c) 2017-2023 TypeART Authors
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

/**
 * Returns whether the pointer was allocated on the device, or for the host.
 * Uses the CUDA runtime API (CU_MEMORYTYPE_DEVICE)
 *
 * \param[in] addr The address.
 * \param[out] is_device_ptr_flag Whether the pointer is on the device (true) or host (false)
 *
 * \return One of the following status codes:
 *  - TYPEART_OK: Success.
 *  - TYPEART_ERROR: The query raised an error in the CUDA runtime
 */
typeart_status typeart_cuda_is_device_ptr(const void* addr, bool* is_device_ptr_flag);

#ifdef __cplusplus
}
#endif

#endif  // TYPEART_CUDARUNTIMEINTERFACE_H

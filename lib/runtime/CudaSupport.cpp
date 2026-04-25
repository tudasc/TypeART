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

#include "CudaRuntimeInterface.h"

#ifdef TYPEART_HAS_CUDA
#include <cuda.h>
#endif

typeart_status typeart_cuda_is_device_ptr(const void* addr, bool* is_device_ptr_flag) {
  if (is_device_ptr_flag == nullptr) {
    return TYPEART_ERROR;
  }

#ifdef TYPEART_HAS_CUDA
  CUmemorytype mem_type;
  CUresult status =
      cuPointerGetAttribute(&mem_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, reinterpret_cast<CUdeviceptr>(addr));
  if (status != CUDA_SUCCESS) {
    *is_device_ptr_flag = false;
    if (status == CUDA_ERROR_INVALID_VALUE) {
      return TYPEART_OK;
    }
    return TYPEART_ERROR;
  }

  *is_device_ptr_flag = (mem_type == CU_MEMORYTYPE_DEVICE);
  return TYPEART_OK;
#else
  (void)addr;
  *is_device_ptr_flag = false;
  return TYPEART_OK;
#endif
}

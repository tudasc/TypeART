#include "RuntimeInterface.h"

#ifdef TYPEART_HAS_CUDA
#include <cuda.h>
#endif

typeart_status typeart_cuda_is_device_ptr(const void* addr, bool* is_device_ptr_flag) {
#ifdef TYPEART_HAS_CUDA
  CUmemorytype mem_type;
  CUresult return_status = cuPointerGetAttribute(&mem_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)addr);
  if (return_status != CUDA_SUCCESS) {
    *is_device_ptr_flag = false;
    if (CUDA_ERROR_INVALID_VALUE == return_status) {
      return TYPEART_OK;
    }
    return TYPEART_ERROR;
  }
  *is_device_ptr_flag = mem_type == CU_MEMORYTYPE_DEVICE;
  return TYPEART_OK;
#else
  *is_device_ptr_flag = false;
  return TYPEART_OK;
#endif
}
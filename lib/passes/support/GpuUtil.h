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

#ifndef TYPEART_GPUUTIL_H
#define TYPEART_GPUUTIL_H

#include "analysis/MemOpData.h"
#include "support/CudaUtil.h"
#include "support/HipUtil.h"

#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#include <optional>

namespace typeart::gpu {

inline bool is_device_module(const llvm::Module& module) {
  return cuda::is_device_module(module) || hip::is_device_module(module);
}

inline std::optional<llvm::BitCastInst*> bitcast_for(const llvm::CallBase& cb, MemOpKind kind) {
  if (kind == MemOpKind::CudaMallocLike) {
    return cuda::bitcast_for(cb);
  }
  if (kind == MemOpKind::HipMallocLike) {
    return hip::bitcast_for(cb);
  }
  return std::nullopt;
}

inline bool is_templated_malloc_like(llvm::StringRef name, MemOpKind kind) {
  if (kind == MemOpKind::CudaMallocLike) {
    return cuda::is_templated_malloc_like(name);
  }
  if (kind == MemOpKind::HipMallocLike) {
    return hip::is_templated_malloc_like(name);
  }
  return false;
}

inline bool is_templated_malloc_like(const llvm::Function& function, MemOpKind kind) {
  if (kind == MemOpKind::CudaMallocLike) {
    return cuda::is_templated_malloc_like(function);
  }
  if (kind == MemOpKind::HipMallocLike) {
    return hip::is_templated_malloc_like(function);
  }
  return false;
}

inline bool is_gpu_function(const llvm::Function& function) {
  return cuda::is_cuda_function(function) || hip::is_hip_function(function);
}

}  // namespace typeart::gpu

#endif  // TYPEART_GPUUTIL_H

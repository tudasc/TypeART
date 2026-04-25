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

#ifndef TYPEART_CUDAUTIL_H
#define TYPEART_CUDAUTIL_H

#include "support/Util.h"

#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#include <optional>
#include <string>

namespace typeart::cuda {

inline std::optional<llvm::BitCastInst*> bitcast_for(llvm::Value* cuda_ptr) {
  std::optional<llvm::BitCastInst*> fallback;
  for (auto& use : cuda_ptr->uses()) {
    auto* use_value = use.get();
    auto* bitcast   = llvm::dyn_cast<llvm::BitCastInst>(use_value);
    if (bitcast == nullptr) {
      continue;
    }

    if (auto* primary_bitcast = llvm::dyn_cast<llvm::BitCastInst>(bitcast->getOperand(0))) {
      return primary_bitcast;
    }

    fallback = bitcast;
    return fallback;
  }
  return fallback;
}

inline std::optional<llvm::BitCastInst*> bitcast_for(const llvm::CallBase& cuda_call) {
  return bitcast_for(cuda_call.getArgOperand(0));
}

inline bool is_device_module(const llvm::Module& module) {
#if LLVM_VERSION_MAJOR >= 20
  const auto triple = module.getTargetTriple().str();
#else
  const auto triple = module.getTargetTriple();
#endif
  return llvm::StringRef{triple}.find("nvptx") != llvm::StringRef::npos;
}

inline bool is_device_stub(const llvm::Function& function) {
  const auto function_name = util::demangle(function.getName());
  return function_name.find("__device_stub__") != std::string::npos;
}

inline bool is_dim3_init(const llvm::Function& function) {
  const auto function_name = util::demangle(function.getName());
  return function_name.find("dim3::dim3") != std::string::npos;
}

inline bool is_cuda_function(const llvm::Function& function) {
  const auto function_name = llvm::StringRef{function.getName()};
  return util::starts_with_any_of(function_name, "cuda");
}

inline bool is_cuda_helper_function(const llvm::Function& function) {
  if (is_device_stub(function) || is_dim3_init(function)) {
    return true;
  }
  const auto function_name = llvm::StringRef{function.getName()};
  return util::starts_with_any_of(function_name, "__cuda");
}

}  // namespace typeart::cuda

#endif  // TYPEART_CUDAUTIL_H

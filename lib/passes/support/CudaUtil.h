// TypeART library
//
// Copyright (c) 2017-2022 TypeART Authors
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

#include "support/Logger.h"
#include "support/TypeUtil.h"
#include "support/Util.h"

#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"

namespace typeart::cuda {

inline llvm::Optional<llvm::BitCastInst*> bitcast_for(llvm::Value* cuda_ptr) {
  // TODO return a vector of bitcasts (with primary (first elem) being the most specific one)
  llvm::BitCastInst* non_primary{nullptr};
  for (auto& use : cuda_ptr->uses()) {
    auto* use_value = use.get();
    if (auto* bitcast = llvm::dyn_cast<llvm::BitCastInst>(use_value)) {
      // If outlined, templatized cudamalloc function is analyzed:
      if (util::type::isVoidPtr(bitcast->getDestTy()) ||
          util::type::isVoidPtr(bitcast->getDestTy()->getPointerElementType())) {
        if (auto* primary_bitcast = llvm::dyn_cast<llvm::BitCastInst>(bitcast->getOperand(0))) {
          return primary_bitcast;
        }
        non_primary = bitcast;
        continue;
      }
      return bitcast;
    }
  }
  return (non_primary == nullptr) ? llvm::None : llvm::Optional{non_primary};
}

inline llvm::Optional<llvm::BitCastInst*> bitcast_for(const llvm::CallBase& cuda_call) {
  return bitcast_for(cuda_call.getArgOperand(0));
}

inline bool is_cuda(llvm::Module& module) {
  const bool is_cuda = module.getTargetTriple().find("nvptx") != std::string::npos;
  return is_cuda;
}

inline bool is_device_stub(llvm::Function& stub_func) {
  const bool is_stub = util::try_demangle(stub_func).find("__device_stub__") != std::string::npos;
  return is_stub;
}

inline bool is_dim3_init(llvm::Function& func) {
  const bool is_dim3_init = util::try_demangle(func).find("dim3::dim3") != std::string::npos;
  return is_dim3_init;
}

}  // namespace typeart::cuda

#endif  // TYPEART_CUDAUTIL_H

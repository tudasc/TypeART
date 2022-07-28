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

#include "TypeUtil.h"
#include "support/Logger.h"

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

}  // namespace typeart::cuda

#endif  // TYPEART_CUDAUTIL_H

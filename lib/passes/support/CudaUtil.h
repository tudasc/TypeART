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

#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"

namespace typeart::cuda {
inline llvm::Optional<llvm::BitCastInst*> bitcast_for(llvm::Value* cuda_ptr) {
  for (auto& use : cuda_ptr->uses()) {
    auto* use_value = use.get();
    if (auto bitcast = llvm::dyn_cast<llvm::BitCastInst>(use_value)) {
      return bitcast;
    }
  }
  return llvm::None;
}

inline llvm::Optional<llvm::BitCastInst*> bitcast_for(const llvm::CallBase& cuda_call) {
  return bitcast_for(cuda_call.getArgOperand(0));
}

}  // namespace typeart::cuda

#endif  // TYPEART_CUDAUTIL_H

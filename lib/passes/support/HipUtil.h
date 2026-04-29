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

#ifndef TYPEART_HIPUTIL_H
#define TYPEART_HIPUTIL_H

#include "analysis/MemOpData.h"
#include "support/Util.h"

#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#include <algorithm>
#include <optional>
#include <string>

namespace typeart::hip {

inline std::optional<llvm::BitCastInst*> bitcast_for(llvm::Value* hip_ptr) {
  std::optional<llvm::BitCastInst*> fallback;
  for (auto& use : hip_ptr->uses()) {
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

inline std::optional<llvm::BitCastInst*> bitcast_for(const llvm::CallBase& hip_call) {
  return bitcast_for(hip_call.getArgOperand(0));
}

inline bool is_device_module(const llvm::Module& module) {
#if LLVM_VERSION_MAJOR >= 20
  const auto triple = module.getTargetTriple().str();
#else
  const auto triple = module.getTargetTriple();
#endif
  return llvm::StringRef{triple}.find("amdgcn") != llvm::StringRef::npos;
}

inline bool is_hip_function(const llvm::Function& function) {
  const auto function_name = util::try_demangle(function);
  return util::starts_with_any_of(function_name, "hip");
}

inline bool is_templated_malloc_like(llvm::StringRef name) {
  const auto templ_start_pos = name.find_first_of('<');
  if (templ_start_pos == llvm::StringRef::npos) {
    return false;
  }
  auto extracted_fn = name.substr(0, templ_start_pos);
  MemOps ops;
  return ops.allocKind(extracted_fn) == MemOpKind::HipMallocLike;
}

inline bool is_templated_malloc_like(const llvm::Function& function) {
  const std::string name = util::try_demangle(function);
  return is_templated_malloc_like(name);
}

}  // namespace typeart::hip

#endif  // TYPEART_HIPUTIL_H

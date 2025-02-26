// TypeART library
//
// Copyright (c) 2017-2025 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

// In Clang 11 CallSite.h was removed, this is a small wrapper reimplementation

#ifndef COMPAT_LLVM_IR_CALLSITE_H
#define COMPAT_LLVM_IR_CALLSITE_H

#include "llvm/IR/Instruction.h"

#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

namespace llvm {
class CallSite {
 private:
  llvm::Instruction* instruction_{nullptr};

 public:
  explicit CallSite(llvm::Instruction* instruction) : instruction_{instruction} {
  }

  explicit CallSite(llvm::Value* instruction) : instruction_{dyn_cast<llvm::Instruction>(instruction)} {
  }

  [[nodiscard]] bool isCall() const {
    return instruction_ != nullptr && llvm::isa<llvm::CallInst>(instruction_);
  }

  [[nodiscard]] bool isInvoke() const {
    return instruction_ != nullptr && llvm::isa<llvm::InvokeInst>(instruction_);
  }

  [[nodiscard]] bool isCallBr() const {
    return instruction_ != nullptr && llvm::isa<llvm::CallBrInst>(instruction_);
  }

  [[nodiscard]] bool isIndirectCall() const {
    if (const CallBase* CB = dyn_cast_or_null<CallBase>(instruction_)) {
      return CB->isIndirectCall();
    }
    return true;
  }

  [[nodiscard]] llvm::Function* getCalledFunction() const {
    if (auto* call_base = llvm::dyn_cast_or_null<llvm::CallBase>(instruction_)) {
      return call_base->getCalledFunction();
    }
    return nullptr;
  }

  [[nodiscard]] auto arg_begin() const {
    auto* call_base = llvm::cast<llvm::CallBase>(instruction_);
    return call_base->arg_begin();
  }

  [[nodiscard]] auto arg_end() const {
    auto* call_base = llvm::cast<llvm::CallBase>(instruction_);
    return call_base->arg_end();
  }

  [[nodiscard]] auto args() const {
    return make_range(arg_begin(), arg_end());
  }

  [[nodiscard]] auto getArgOperand(unsigned int number) const {
    auto* call_base = llvm::cast<llvm::CallBase>(instruction_);
    return call_base->getArgOperand(number);
  }

  auto getIntrinsicID() const {
    auto* call_base = llvm::cast<llvm::CallBase>(instruction_);
    return call_base->getIntrinsicID();
  }
};
}  // namespace llvm

#endif

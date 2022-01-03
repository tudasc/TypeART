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

#ifndef TYPEART_FILTERUTIL_H
#define TYPEART_FILTERUTIL_H

#include "IRPath.h"
#include "OmpUtil.h"
#include "compat/CallSite.h"
#include "support/DefUseChain.h"
#include "support/Logger.h"

#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

namespace llvm {
class Value;
class raw_ostream;
}  // namespace llvm

using namespace llvm;

namespace typeart::filter {

struct FunctionAnalysis {
  using FunctionCounts = struct { int decl, def, intrinsic, indirect; };
  using FunctionCalls  = struct { llvm::SmallVector<CallSite, 8> decl, def, intrinsic, indirect; };

  FunctionCalls calls;

  void clear();

  bool empty() const;

  FunctionCounts analyze(Function* f);
};

raw_ostream& operator<<(raw_ostream& os, const FunctionAnalysis::FunctionCounts& counts);

enum class ArgCorrelation {
  NoMatch,
  Exact,
  ExactMismatch,
  Global,
  GlobalMismatch,
};

inline std::pair<llvm::Argument*, int> findArg(CallSite c, const Path& p) {
  auto arg = p.getEndPrev();
  if (!arg) {
    return {nullptr, -1};
  }

  Value* in          = arg.getValue();
  const auto arg_pos = llvm::find_if(c.args(), [&in](const Use& arg_use) -> bool { return arg_use.get() == in; });

  if (arg_pos == c.arg_end()) {
    return {nullptr, -1};
  }

  auto arg_num = std::distance(c.arg_begin(), arg_pos);

  if (omp::OmpContext::isOmpExecutor(c)) {
    auto outlined = omp::OmpContext::getMicrotask(c);
    if (outlined) {
      // Calc the offset of arg in executor to actual arg of the outline function:
      auto offset        = omp::OmpContext::getArgOffsetToMicrotask(c, arg_num);
      Argument* argument = (outlined.getValue()->arg_begin() + offset);
      return {argument, offset};
    }
  }

  Argument* argument = c.getCalledFunction()->arg_begin() + arg_num;
  return {argument, arg_num};
}

inline std::vector<llvm::Argument*> args(CallSite c, const Path& p) {
  if (c.isIndirectCall()) {
    return {};
  }

  auto [arg, _] = findArg(c, p);
  if (arg != nullptr) {
    return {arg};
  }

  std::vector<llvm::Argument*> args;
  llvm::for_each(c.getCalledFunction()->args(), [&](llvm::Argument& a) { args.emplace_back(&a); });
  return args;
}

namespace detail {
template <typename TypeID>
ArgCorrelation correlate(CallSite c, const Path& p, TypeID&& isType) {
  auto [arg, _] = findArg(c, p);

  if (!arg) {
    const auto count_type_ptr = llvm::count_if(c.args(), [&](const auto& csite_arg) {
      const auto type = csite_arg->getType();
      return isType(type);
    });
    if (count_type_ptr > 0) {
      return ArgCorrelation::Global;
    }
    return ArgCorrelation::GlobalMismatch;
  }

  auto type = arg->getType();

  if (isType(type)) {
    return ArgCorrelation::Exact;
  }
  return ArgCorrelation::ExactMismatch;
}
}  // namespace detail

inline ArgCorrelation correlate2void(CallSite c, const Path& p) {
  return detail::correlate(
      c, p, [](llvm::Type* type) { return type->isPointerTy() && type->getPointerElementType()->isIntegerTy(8); });
}

inline ArgCorrelation correlate2pointer(CallSite c, const Path& p) {
  // weaker predicate than void pointer, but more generally applicable
  return detail::correlate(c, p, [](llvm::Type* type) { return type->isPointerTy(); });
}

inline bool isTempAlloc(llvm::Value* in) {
  const auto farg_stored_to = [](llvm::AllocaInst* inst) -> bool {
    bool match{false};
    Function* f = inst->getFunction();

    util::DefUseChain chain;
    chain.traverse(inst, [&f, &match](auto val) {
      if (llvm::StoreInst* store = llvm::dyn_cast<StoreInst>(val)) {
        for (auto& args : f->args()) {
          if (&args == store->getValueOperand()) {
            match = true;
            return util::DefUseChain::cancel;
          }
        }
      }
      return util::DefUseChain::no_match;
    });

    return match;
  };
  if (llvm::AllocaInst* inst = llvm::dyn_cast<llvm::AllocaInst>(in)) {
    if (inst->getAllocatedType()->isPointerTy()) {
      return farg_stored_to(inst);
    }
  }
  return false;
}

}  // namespace typeart::filter

#endif  // TYPEART_FILTERUTIL_H

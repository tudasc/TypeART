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

#ifndef TYPEART_FILTER_OMPUTIL_H
#define TYPEART_FILTER_OMPUTIL_H

#include "IRPath.h"
#include "compat/CallSite.h"
#include "support/DefUseChain.h"
#include "support/OmpUtil.h"
#include "support/Util.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/Casting.h"

#include <optional>

namespace typeart::filter::omp {
struct EmptyContext {
  constexpr static bool WithOmp = false;
};

struct OmpContext {
  constexpr static bool WithOmp = true;

  static bool isOmpExecutor(const llvm::CallSite& c) {
    const auto called = c.getCalledFunction();
    if (called != nullptr) {
      // TODO probably not complete (openmp task?, see isOmpTask*())
      return util::starts_with_any_of(called->getName(), "__kmpc_fork_call");
    }
    return false;
  }

  static bool isOmpTaskAlloc(const llvm::CallSite& c) {
    const auto called = c.getCalledFunction();
    if (called != nullptr) {
      return util::starts_with_any_of(called->getName(), "__kmpc_omp_task_alloc");
    }
    return false;
  }

  static bool isOmpTaskCall(const llvm::CallSite& c) {
    const auto called = c.getCalledFunction();
    if (called != nullptr) {
      return util::ends_with_any_of(called->getName(), "__kmpc_omp_task");
    }
    return false;
  }

  static bool isOmpTaskRelated(const llvm::CallSite& c) {
    const auto called = c.getCalledFunction();
    if (called != nullptr) {
      return util::starts_with_any_of(called->getName(), "__kmpc_omp_task");
    }
    return false;
  }

  static bool isOmpHelper(const llvm::CallSite& c) {
    const auto is_execute = isOmpExecutor(c);
    if (!is_execute) {
      const auto called = c.getCalledFunction();
      if (called != nullptr) {
        // TODO extend this if required
        return util::starts_with_any_of(called->getName(), "__kmpc", "omp_");
      }
    }
    return false;
  }

  static std::optional<llvm::Function*> getMicrotask(const llvm::CallSite& c) {
    using namespace llvm;
    if (isOmpExecutor(c)) {
      auto f = llvm::dyn_cast<llvm::Function>(c.getArgOperand(2)->stripPointerCasts());
      return {f};
    }
    if (isOmpTaskAlloc(c)) {
      auto f = llvm::dyn_cast<llvm::Function>(c.getArgOperand(5)->stripPointerCasts());
      return {f};
    }
    return {};
  }

  static bool canDiscardMicrotaskArg(llvm::CallSite c, const Path& path) {
    using namespace llvm;
    auto arg = path.getEndPrev();
    if (!arg) {
      return false;
    }

    Value* in          = arg.value();
    const auto arg_pos = llvm::find_if(c.args(), [&in](const Use& arg_use) -> bool { return arg_use.get() == in; });

    if (arg_pos == c.arg_end()) {
      return false;
    }

    auto arg_num = std::distance(c.arg_begin(), arg_pos);

    if (isOmpExecutor(c)) {
      return arg_num <= 2;
    }

    if (isOmpTaskAlloc(c)) {
      return arg_num <= 5;  // task alloc inits the task, discard all 5
    }

    if (isOmpTaskCall(c)) {
      return arg_num <= 2;  // task call executes, in theory, discard only first 2
    }

    return false;
  }

  static bool allocaReachesTask(llvm::AllocaInst* alloc) {
    if (!util::omp::isOmpContext(alloc->getFunction())) {
      return false;
    }

    bool found{false};
    util::DefUseChain finder;
    finder.traverse_custom(
        alloc,
        [](auto val) -> std::optional<decltype(val->users())> {
          if (auto cinst = llvm::dyn_cast<llvm::StoreInst>(val)) {
            return cinst->getValueOperand()->users();
          }
          return val->users();
        },
        [&found](auto value) {
          llvm::CallSite site(value);
          if (site.isCall() || site.isInvoke()) {
            const auto called = site.getCalledFunction();

            if (called != nullptr && util::starts_with_any_of(called->getName(), "__kmpc_omp_task(")) {
              found = true;
              return util::DefUseChain::kCancel;
            }
          }
          return util::DefUseChain::kNoMatch;
        });
    return found;
  }

  static bool isTaskRelatedStore(llvm::Value* v) {
    if (llvm::StoreInst* store = llvm::dyn_cast<llvm::StoreInst>(v)) {
      llvm::Function* f = store->getFunction();
      if (util::omp::isOmpContext(f)) {
        auto operand = store->getPointerOperand();
        if (llvm::GEPOperator* gep = llvm::dyn_cast<llvm::GEPOperator>(operand)) {
          auto type = gep->getSourceElementType();
          // Second condition filters out many struct.ident_t in lulesh omp:
          if (llvm::isa<llvm::StructType>(type) && !type->getStructName().contains("struct.ident_t")) {
            //            LOG_FATAL(*(gep->getSourceElementType()))
            return true;
          }
        }
        // else find task_alloc, and correlate with store (arg "v") to result of task_alloc
        auto calls = util::find_all(f, [&](auto& inst) {
          llvm::CallSite s(&inst);
          if (s.isCall() || s.isInvoke()) {
            if (auto called_function = s.getCalledFunction()) {
              // once true, the find_all should cancel
              return util::starts_with_any_of(called_function->getName(), "__kmpc_omp_task_alloc");
            }
          }
          return false;
        });

        bool found{false};
        util::DefUseChain chain;
        for (auto i : calls) {
          chain.traverse(i, [&v, &found](auto val) {
            if (v == val) {
              found = true;
              return util::DefUseChain::kCancel;
            }
            return util::DefUseChain::kNoMatch;
          });
          if (found) {
            return true;
          }
        }
      }
    }

    return false;
  }

  template <typename Distance>
  static Distance getArgOffsetToMicrotask(const llvm::CallSite& c, Distance d) {
    if (d < 1) {
      LOG_WARNING("OMP offset should be > 2 for non-omp-internal args to outlined region")
      return d;
    }
    if (isOmpExecutor(c)) {
      return d - Distance{1};
    }
    LOG_WARNING("Unsupported OMP call.")
    return d;
  }
};

}  // namespace typeart::filter::omp

#endif  // TYPEART_OMPUTIL_H

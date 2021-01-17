//
// Created by ahueck on 17.01.21.
//

#ifndef TYPEART_OMPUTIL_H
#define TYPEART_OMPUTIL_H

#include "llvm/ADT/Optional.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Function.h"

namespace typeart::filter::thread {

struct EmptyContext {
  constexpr static bool WithOmp = false;
};

struct OmpContext {
  constexpr static bool WithOmp = true;

  static bool isOmpExecutor(const llvm::CallSite& c) {
    const auto called = c.getCalledFunction();
    if (called != nullptr) {
      // TODO probably not complete (openmp task?)
      return called->getName().startswith("__kmpc_fork_call");
    }
    return false;
  }

  static bool isOmpHelper(const llvm::CallSite& c) {
    const auto is_execute = isOmpExecutor(c);
    if (!is_execute) {
      const auto called = c.getCalledFunction();
      if (called != nullptr) {
        const auto name = called->getName();
        // TODO extend this if required
        return name.startswith("__kmpc") || name.startswith("omp_");
      }
    }
    return false;
  }

  static llvm::Optional<llvm::Function*> getMicrotask(const llvm::CallSite& c) {
    using namespace llvm;
    if (isOmpExecutor(c)) {
      auto f = llvm::dyn_cast<llvm::Function>(c.getArgOperand(2)->stripPointerCasts());
      return {f};
    }
    return llvm::None;
  }

  template <typename Distance>
  static Distance getArgOffsetToMicrotask(Distance d) {
    return d - Distance{1};
  }
};

}  // namespace typeart::filter::thread

#endif  // TYPEART_OMPUTIL_H

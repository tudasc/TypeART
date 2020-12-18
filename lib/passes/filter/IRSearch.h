//
// Created by ahueck on 06.11.20.
//

#ifndef TYPEART_IRSEARCH_H
#define TYPEART_IRSEARCH_H

#include "llvm/IR/CallSite.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"

namespace typeart::filter {

struct DefaultSearch {
  auto search(llvm::Value* val, const Path& p) -> std::vector<llvm::Value*> {
    using namespace llvm;

    std::vector<llvm::Value*> out;

    if (auto store = llvm::dyn_cast<StoreInst>(val)) {
      val = store->getPointerOperand();
      if (llvm::isa<AllocaInst>(val) && !store->getValueOperand()->getType()->isPointerTy()) {
        // 1. if we store to an alloca, and the value is not a pointer (i.e., a value) there is no connection to follow
        // w.r.t. dataflow. (TODO exceptions could be some pointer arithm.)
        return {};
      }
      if (p.contains(val)) {
        // If the pointer operand is already in the path, we do not want to continue.
        // Encountered: amg: box_algebra.c: hypre_MinUnionBoxes endless recursion
        return {};
      }

      out.push_back(val);
      // The following return is needed to fix endless recursions (see test/pass/filter/ -> 08 and 09)
      return out;  // this is what we need in case of following store target

      // 2. TODO if we store to a pointer, analysis is required to filter simple aliasing pointer (filter opportunity,
      // see test 01_alloca.llin variable a and c -- c points to a, then c gets passed to MPI)
      // 2.1 care has to be taken for argument store to aliasing local (implicit) alloc, i.e., see same test variable %x
      // passed to func foo_bar
    }

    llvm::transform(val->users(), std::back_inserter(out), [](User* u) { return dyn_cast<Value>(u); });
    return out;
  }
};

}  // namespace typeart::filter

#endif  // TYPEART_IRSEARCH_H

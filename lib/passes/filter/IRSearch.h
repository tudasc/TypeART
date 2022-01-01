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

#ifndef TYPEART_IRSEARCH_H
#define TYPEART_IRSEARCH_H

#include "compat/CallSite.h"

#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"

namespace typeart::filter {

struct DefaultSearch {
  auto search(llvm::Value* val, const Path& p) -> std::vector<llvm::Value*> {
    std::vector<llvm::Value*> out;

    if (isa<llvm::PHINode>(val)) {
      // FIXME
      //  this mechanism tries to avoid endless recurison in loops, i.e.,
      //  do we bounce around multiple phi nodes (visit counter >1), then
      //  we should likely skip search
      // see amg with openmp `amg2013/parcsr_ls/par_lr_interp.c`
      if (const auto node = p.phi_cache.find(val); node != std::end(p.phi_cache)) {
        if (node->second > 1) {
          return out;
        }
      }
    }

    if (auto store = llvm::dyn_cast<llvm::StoreInst>(val)) {
      val = store->getPointerOperand();
      if (llvm::isa<AllocaInst>(val) && !store->getValueOperand()->getType()->isPointerTy()) {
        // 1. if we store to an alloca, and the value is not a pointer (i.e., a value) there is no connection to follow
        // w.r.t. dataflow. (TODO exceptions could be some pointer arithm.)
        return out;
      }
      if (p.contains(val)) {
        // If the pointer operand is already in the path, we do not want to continue.
        // Encountered: amg: box_algebra.c: hypre_MinUnionBoxes endless recursion
        return out;
      }

      out.push_back(val);
      // The following return is needed to fix endless recursions (see test/pass/filter/ -> 08 and 09)
      return out;  // this is what we need in case of following store target

      // 2. TODO if we store to a pointer, analysis is required to filter simple aliasing pointer (filter opportunity,
      // see test 01_alloca.llin variable a and c -- c points to a, then c gets passed to MPI)
      // 2.1 care has to be taken for argument store to aliasing local (implicit) alloc, i.e., see same test variable %x
      // passed to func foo_bar
    }

    llvm::transform(val->users(), std::back_inserter(out), [](llvm::User* u) { return dyn_cast<llvm::Value>(u); });
    return out;
  }
};

}  // namespace typeart::filter

#endif  // TYPEART_IRSEARCH_H

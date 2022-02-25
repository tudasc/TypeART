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

#ifndef TYPEART_TRANSFORMUTIL_H
#define TYPEART_TRANSFORMUTIL_H

#include "InstrumentationHelper.h"
#include "TypeARTFunctions.h"
#include "support/OmpUtil.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"

namespace typeart::transform {

struct StackCounter {
  using StackOpCounter = llvm::SmallDenseMap<llvm::BasicBlock*, size_t>;
  llvm::Function* f;
  InstrumentationHelper* instr_helper;
  TAFunctionQuery* fquery;

  StackCounter(llvm::Function* f, InstrumentationHelper* instr, TAFunctionQuery* query)
      : f(f), instr_helper(instr), fquery(query) {
  }

  void addStackHandling(StackOpCounter& allocCounts) const {
    using namespace llvm;
    //      LOG_DEBUG("Add alloca counter")
    // counter = 0 at beginning of function
    IRBuilder<> CBuilder(f->getEntryBlock().getFirstNonPHI());
    auto* counter = CBuilder.CreateAlloca(instr_helper->getTypeFor(IType::stack_count), nullptr, "__ta_alloca_counter");
    CBuilder.CreateStore(instr_helper->getConstantFor(IType::stack_count), counter);

    // In each basic block: counter =+ num_alloca (in BB)
    for (auto data : allocCounts) {
      IRBuilder<> IRB(data.first->getTerminator());
      auto* load_counter = IRB.CreateLoad(instr_helper->getTypeFor(IType::stack_count), counter);
      Value* increment_counter =
          IRB.CreateAdd(instr_helper->getConstantFor(IType::stack_count, data.second), load_counter);
      IRB.CreateStore(increment_counter, counter);
    }

    // Find return instructions:
    // if(counter > 0) call runtime for stack cleanup
    const auto callback_id = util::omp::isOmpContext(f) ? IFunc::scope_omp : IFunc::scope;

    EscapeEnumerator ee(*f);
    while (IRBuilder<>* irb = ee.Next()) {
      auto* I            = &(*irb->GetInsertPoint());
      auto* counter_load = irb->CreateLoad(instr_helper->getTypeFor(IType::stack_count), counter, "__ta_counter_load");

      const auto all_preds_have_counter = llvm::all_of(
          llvm::predecessors(I->getParent()), [&allocCounts](const auto* bb) { return allocCounts.count(bb) > 0; });
      if (all_preds_have_counter) {
        auto* cond = irb->CreateICmpNE(counter_load, instr_helper->getConstantFor(IType::stack_count), "__ta_cond");
        auto* then_term = SplitBlockAndInsertIfThen(cond, I, false);
        irb->SetInsertPoint(then_term);
      }

      irb->CreateCall(fquery->getFunctionFor(callback_id), ArrayRef<Value*>{counter_load});
    }
  }
};

}  // namespace typeart::transform

#endif  // TYPEART_TRANSFORMUTIL_H

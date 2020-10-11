//
// Created by ahueck on 09.10.20.
//

#ifndef TYPEART_TRANSFORMUTIL_H
#define TYPEART_TRANSFORMUTIL_H

#include "InstrumentationHelper.h"
#include "TypeARTFunctions.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"

namespace typeart {
namespace transform {

struct StackCounter {
  using StackOpCounter = llvm::SmallDenseMap<llvm::BasicBlock*, size_t>;
  llvm::Function* f;
  InstrumentationHelper& instr;
  TAFunctionQuery& query;

  StackCounter(llvm::Function* f, InstrumentationHelper& instr, TAFunctionQuery& query)
      : f(f), instr(instr), query(query) {
  }

  void addStackHandling(StackOpCounter& allocCounts) {
    using namespace llvm;
    //      LOG_DEBUG("Add alloca counter")
    // counter = 0 at beginning of function
    IRBuilder<> CBuilder(f->getEntryBlock().getFirstNonPHI());
    auto* counter = CBuilder.CreateAlloca(instr.getTypeFor(IType::stack_count), nullptr, "__ta_alloca_counter");
    CBuilder.CreateStore(instr.getConstantFor(IType::stack_count), counter);

    // In each basic block: counter =+ num_alloca (in BB)
    for (auto data : allocCounts) {
      IRBuilder<> IRB(data.first->getTerminator());
      auto* load_counter       = IRB.CreateLoad(counter);
      Value* increment_counter = IRB.CreateAdd(instr.getConstantFor(IType::stack_count, data.second), load_counter);
      IRB.CreateStore(increment_counter, counter);
    }

    // Find return instructions:
    // if(counter > 0) call runtime for stack cleanup
    EscapeEnumerator ee(*f);
    while (IRBuilder<>* irb = ee.Next()) {
      auto* I = &(*irb->GetInsertPoint());

      auto* counter_load = irb->CreateLoad(counter, "__ta_counter_load");
      auto* cond         = irb->CreateICmpNE(counter_load, instr.getConstantFor(IType::stack_count), "__ta_cond");
      auto* then_term    = SplitBlockAndInsertIfThen(cond, I, false);
      irb->SetInsertPoint(then_term);
      irb->CreateCall(query.getFunctionFor(IFunc::scope), ArrayRef<Value*>{counter_load});
    }
  }
};

}  // namespace transform
}  // namespace typeart

#endif  // TYPEART_TRANSFORMUTIL_H

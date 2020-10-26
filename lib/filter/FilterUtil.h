//
// Created by ahueck on 26.10.20.
//

#ifndef TYPEART_FILTERUTIL_H
#define TYPEART_FILTERUTIL_H

#include "llvm/IR/CallSite.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"

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

struct DefUseQueue {
  llvm::SmallPtrSet<Value*, 16> visited_set;
  llvm::SmallVector<Value*, 16> working_set;
  llvm::SmallVector<CallSite, 8> working_set_calls;

  explicit DefUseQueue(Value* init);

  void reset();

  bool empty() const;

  void addToWorkS(Value* v);

  template <typename Range>
  void addToWork(Range&& values) {
    for (auto v : values) {
      addToWorkS(v);
    }
  }

  Value* peek();
};

}  // namespace typeart::filter

#endif  // TYPEART_FILTERUTIL_H

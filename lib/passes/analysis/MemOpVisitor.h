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

#ifndef LIB_MEMOPVISITOR_H_
#define LIB_MEMOPVISITOR_H_

#include "MemOpData.h"

#include "llvm/IR/InstVisitor.h"

namespace llvm {
class AllocaInst;
class CallBase;
class Module;
class InstrinsicInst;
class Function;
}  // namespace llvm

namespace typeart::analysis {

struct MemOpVisitor : public llvm::InstVisitor<MemOpVisitor> {
  GlobalDataList globals;
  MallocDataList mallocs;
  FreeDataList frees;
  AllocaDataList allocas;
  llvm::SmallVector<std::pair<llvm::IntrinsicInst*, llvm::AllocaInst*>, 16> lifetime_starts;

 private:
  MemOps mem_operations{};
  bool collect_allocas;
  bool collect_heap;

 public:
  MemOpVisitor();
  MemOpVisitor(bool collect_allocas, bool collect_heap);
  void collect(llvm::Function& function);
  void collectGlobals(llvm::Module& module);
  void clear();

  void visitCallBase(llvm::CallBase& cb);
  void visitMallocLike(llvm::CallBase& ci, MemOpKind k);
  void visitFreeLike(llvm::CallBase& ci, MemOpKind k);
  void visitAllocaInst(llvm::AllocaInst& ai);
  void visitIntrinsicInst(llvm::IntrinsicInst& inst);
};

}  // namespace typeart::analysis

#endif /* LIB_MEMOPVISITOR_H_ */

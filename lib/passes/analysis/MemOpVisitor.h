// TypeART library
//
// Copyright (c) 2017-2021 TypeART Authors
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
}  // namespace llvm

namespace typeart::analysis {

struct MemOpVisitor : public llvm::InstVisitor<MemOpVisitor> {
  GlobalDataList globals;
  MallocDataList mallocs;
  FreeDataList frees;
  AllocaDataList allocas;

 private:
  MemOps mem_operations{};
  bool collectAllocas;
  bool collectHeap;

 public:
  MemOpVisitor();
  MemOpVisitor(bool collectAllocas, bool collectHeap);

 public:
  void clear();
  void visitModuleGlobals(llvm::Module& m);
  void visitCallBase(llvm::CallBase& cb);
  void visitMallocLike(llvm::CallBase& ci, MemOpKind k);
  void visitFreeLike(llvm::CallBase& ci, MemOpKind k);
  void visitAllocaInst(llvm::AllocaInst& ai);
};

}  // namespace typeart::analysis

#endif /* LIB_MEMOPVISITOR_H_ */

/*
 * MemOpVisitor.h
 *
 *  Created on: Jan 3, 2018
 *      Author: ahueck
 */

#ifndef LIB_MEMOPVISITOR_H_
#define LIB_MEMOPVISITOR_H_

#include "MemOpData.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/IR/InstVisitor.h"

namespace typeart::finder {

struct MemOpVisitor : public llvm::InstVisitor<MemOpVisitor> {
  GlobalDataList globals;
  MallocDataList mallocs;
  FreeDataList frees;
  AllocaDataList allocas;

 private:
  MemOps mem_operations{};

 public:
  void clear();
  void visitModuleGlobals(llvm::Module& m);
  void visitCallBase(llvm::CallBase& cb);
  void visitMallocLike(llvm::CallBase& ci, MemOpKind k);
  void visitFreeLike(llvm::CallBase& ci, MemOpKind k);
  void visitAllocaInst(llvm::AllocaInst& ai);
};

}  // namespace typeart::finder

#endif /* LIB_MEMOPVISITOR_H_ */

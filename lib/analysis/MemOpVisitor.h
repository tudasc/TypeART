/*
 * MemOpVisitor.h
 *
 *  Created on: Jan 3, 2018
 *      Author: ahueck
 */

#ifndef LIB_MEMOPVISITOR_H_
#define LIB_MEMOPVISITOR_H_

#include "llvm/IR/InstVisitor.h"

#include <set>

namespace typeart {

struct MallocData {
  llvm::CallInst* call{nullptr};
  llvm::BitCastInst* primary{nullptr};  // Non-null if non (void*) cast exists
  llvm::SmallPtrSet<llvm::BitCastInst*, 4> bitcasts;
};

struct MemOpVisitor : public llvm::InstVisitor<MemOpVisitor> {
  MemOpVisitor();
  void clear();
  void visitCallInst(llvm::CallInst& ci);
  void visitMallocLike(llvm::CallInst& ci);
  void visitFreeLike(llvm::CallInst& ci);
  void visitAllocaInst(llvm::AllocaInst& ai);
  virtual ~MemOpVisitor();

  llvm::SmallVector<MallocData, 8> listMalloc;
  llvm::SmallPtrSet<llvm::CallInst*, 8> listFree;
  llvm::SmallPtrSet<llvm::AllocaInst*, 8> listAlloca;

 private:
  /** Look up sets for keyword strings */
  const std::set<std::string> allocFunctions{"malloc", "_Znwm" /*new*/, "_Znam" /*new[]*/};
  const std::set<std::string> deallocFunctions{"free", "_ZdlPv" /*delete*/, "_ZdaPv" /*delete[]*/};
};

}  // namespace typeart

#endif /* LIB_MEMOPVISITOR_H_ */

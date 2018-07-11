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

enum class MemOpKind { MALLOC, CALLOC, REALLOC, FREE };

struct MallocData {
  llvm::CallInst* call{nullptr};
  llvm::BitCastInst* primary{nullptr};  // Non-null if non (void*) cast exists
  llvm::SmallPtrSet<llvm::BitCastInst*, 4> bitcasts;
  MemOpKind kind;
};

struct MemOpVisitor : public llvm::InstVisitor<MemOpVisitor> {
  MemOpVisitor();
  void clear();
  void visitCallInst(llvm::CallInst& ci);
  void visitMallocLike(llvm::CallInst& ci, MemOpKind k);
  void visitFreeLike(llvm::CallInst& ci, MemOpKind k);
  void visitAllocaInst(llvm::AllocaInst& ai);
  virtual ~MemOpVisitor();

  llvm::SmallVector<MallocData, 8> listMalloc;
  llvm::SmallPtrSet<llvm::CallInst*, 8> listFree;
  llvm::SmallPtrSet<llvm::AllocaInst*, 8> listAlloca;

  using MemFuncT = std::pair<std::string, MemOpKind>;

 private:
  /** Look up sets for keyword strings */
  const std::set<MemFuncT> allocFunctions{{"malloc", MemOpKind::MALLOC},
                                          {"_Znwm", MemOpKind::MALLOC} /*new*/,
                                          {"_Znam", MemOpKind::MALLOC} /*new[]*/,
                                          {"calloc", MemOpKind::CALLOC},
                                          {"realloc", MemOpKind::REALLOC}};
  const std::set<MemFuncT> deallocFunctions{
      {"free", MemOpKind::FREE}, {"_ZdlPv", MemOpKind::FREE} /*delete*/, {"_ZdaPv", MemOpKind::FREE} /*delete[]*/};
};

}  // namespace typeart

#endif /* LIB_MEMOPVISITOR_H_ */

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

struct AllocaData {
  llvm::AllocaInst* alloca{nullptr};
  int arraySize; // Number of allocated elements (negative value for VLAs)
};

namespace finder {

struct MemOpVisitor : public llvm::InstVisitor<MemOpVisitor> {
  MemOpVisitor();
  void clear();
  void visitModuleGlobals(llvm::Module& m);
  void visitCallInst(llvm::CallInst& ci);
  void visitMallocLike(llvm::CallInst& ci, MemOpKind k);
  void visitFreeLike(llvm::CallInst& ci, MemOpKind k);
  //  void visitIntrinsicInst(llvm::IntrinsicInst& ii);
  void visitAllocaInst(llvm::AllocaInst& ai);
  virtual ~MemOpVisitor();

  llvm::SmallVector<llvm::GlobalVariable*, 8> listGlobals;
  llvm::SmallVector<MallocData, 8> listMalloc;
  llvm::SmallPtrSet<llvm::CallInst*, 8> listFree;
  llvm::SmallVector<AllocaData, 8> listAlloca;

 private:
  // clang-format off
  const std::map<std::string, MemOpKind> allocMap{{"malloc", MemOpKind::MALLOC},
                                                  {"_Znwm", MemOpKind::MALLOC}, /*new*/
                                                  {"_Znam", MemOpKind::MALLOC}, /*new[]*/
                                                  {"calloc", MemOpKind::CALLOC},
                                                  {"realloc", MemOpKind::REALLOC}
                                                 };
  const std::map<std::string, MemOpKind> deallocMap{{"free", MemOpKind::FREE},
                                                    {"_ZdlPv", MemOpKind::FREE}, /*delete*/
                                                    {"_ZdaPv", MemOpKind::FREE} /*delete[]*/
                                                   };
  // clang-format on
};

}  // namespace finder
}  // namespace typeart

#endif /* LIB_MEMOPVISITOR_H_ */

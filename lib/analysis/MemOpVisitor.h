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
  llvm::CallBase* call{nullptr};
  llvm::BitCastInst* primary{nullptr};  // Non-null if non (void*) cast exists
  llvm::SmallPtrSet<llvm::BitCastInst*, 4> bitcasts;
  MemOpKind kind;
  bool is_invoke{false};
};

struct FreeData {
  llvm::CallBase* call{nullptr};
  bool is_invoke{false};
};

struct AllocaData {
  llvm::AllocaInst* alloca{nullptr};
  size_t array_size;
  bool is_vla{false};
};

struct GlobalData {
  llvm::GlobalVariable* global{nullptr};
};

namespace finder {

struct MemOpVisitor : public llvm::InstVisitor<MemOpVisitor> {
  MemOpVisitor();
  void clear();
  void visitModuleGlobals(llvm::Module& m);
  void visitCallBase(llvm::CallBase& cb);
  void visitMallocLike(llvm::CallBase& ci, MemOpKind k);
  void visitFreeLike(llvm::CallBase& ci, MemOpKind k);
  //  void visitIntrinsicInst(llvm::IntrinsicInst& ii);
  void visitAllocaInst(llvm::AllocaInst& ai);
  virtual ~MemOpVisitor();

  llvm::SmallVector<GlobalData, 8> globals;
  llvm::SmallVector<MallocData, 8> mallocs;
  llvm::SmallVector<FreeData, 8> frees;
  llvm::SmallVector<AllocaData, 8> allocas;

 private:
  // clang-format off
  const std::map<std::string, MemOpKind> alloc_map{{"malloc", MemOpKind::MALLOC},
                                                  {"_Znwm", MemOpKind::MALLOC}, /*new*/
                                                  {"_Znam", MemOpKind::MALLOC}, /*new[]*/
                                                  {"calloc", MemOpKind::CALLOC},
                                                  {"realloc", MemOpKind::REALLOC}
                                                 };
  const std::map<std::string, MemOpKind> dealloc_map{{"free", MemOpKind::FREE},
                                                    {"_ZdlPv", MemOpKind::FREE}, /*delete*/
                                                    {"_ZdaPv", MemOpKind::FREE} /*delete[]*/
                                                   };
  // clang-format on
};

}  // namespace finder
}  // namespace typeart

#endif /* LIB_MEMOPVISITOR_H_ */

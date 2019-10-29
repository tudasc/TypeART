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
enum class AssertKind {TYPE, TYPELEN};
struct MallocData {
  llvm::CallInst* call{nullptr};
  llvm::BitCastInst* primary{nullptr};  // Non-null if non (void*) cast exists
  llvm::SmallPtrSet<llvm::BitCastInst*, 4> bitcasts;
  MemOpKind kind;
};

struct AllocaData {
  llvm::AllocaInst* alloca{nullptr};
  int arraySize;  // Number of allocated elements (negative value for VLAs)
};

struct AssertData {
  llvm::CallInst* call{nullptr};
  AssertKind kind;
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
  void visitTypeAssert(llvm::CallInst& ci, AssertKind k);
  virtual ~MemOpVisitor();

  llvm::SmallVector<llvm::GlobalVariable*, 8> listGlobals;
  llvm::SmallVector<MallocData, 8> listMalloc;
  llvm::SmallPtrSet<llvm::CallInst*, 8> listFree;
  llvm::SmallVector<AllocaData, 8> listAlloca;
  llvm::SmallVector<AssertData, 8> listAssert;

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
  //const std::string assertFuncName{"__typeart_assert_type_stub"};
  const std::map<std::string, AssertKind> assertMap{{"__typeart_assert_type_stub", AssertKind::TYPE},
                                                   {"__typeart_assert_type_stub_len", AssertKind::TYPELEN}
                                                  };
  // clang-format on
};

}  // namespace finder
}  // namespace typeart

#endif /* LIB_MEMOPVISITOR_H_ */

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

namespace typeart {

namespace finder {

struct MemOpVisitor : public llvm::InstVisitor<MemOpVisitor> {
  GlobalDataList globals;
  MallocDataList mallocs;
  FreeDataList frees;
  AllocaDataList allocas;

 private:
  // clang-format off
  const llvm::StringMap<MemOpKind> alloc_map{  {"malloc", MemOpKind::MALLOC},
                                               {"_Znwm",  MemOpKind::NEW}, /*new(unsigned long)*/
                                               {"_Znwj",  MemOpKind::NEW}, /*new(unsigned int)*/
                                               {"_Znam",  MemOpKind::NEW}, /*new[](unsigned long)*/
                                               {"_Znaj",  MemOpKind::NEW}, /*new[](unsigned int)*/
                                               {"_ZnwjRKSt9nothrow_t",  MemOpKind::NEW}, /*new(unsigned int, nothrow)*/
                                               {"_ZnwmRKSt9nothrow_t",  MemOpKind::NEW}, /*new(unsigned long, nothrow)*/
                                               {"_ZnajRKSt9nothrow_t",  MemOpKind::NEW}, /*new[](unsigned int, nothrow)*/
                                               {"_ZnamRKSt9nothrow_t",  MemOpKind::NEW}, /*new[](unsigned long, nothrow)*/
                                               {"calloc", MemOpKind::CALLOC},
                                               {"realloc",MemOpKind::REALLOC}
  };
  const llvm::StringMap<MemOpKind> dealloc_map{{"free",   MemOpKind::FREE},
                                               {"_ZdlPv", MemOpKind::DELETE}, /*delete(void*)*/
                                               {"_ZdaPv", MemOpKind::DELETE}, /*delete[](void*)*/
                                               {"_ZdlPvj", MemOpKind::DELETE}, /*delete(void*, uint)*/
                                               {"_ZdlPvm", MemOpKind::DELETE}, /*delete(void*, ulong)*/
                                               {"_ZdlPvRKSt9nothrow_t", MemOpKind::DELETE}, /*delete(void*, nothrow)*/
                                               {"_ZdaPvj", MemOpKind::DELETE}, /*delete[](void*, uint)*/
                                               {"_ZdaPvm", MemOpKind::DELETE}, /*delete[](void*, ulong)*/
                                               {"_ZdaPvRKSt9nothrow_t", MemOpKind::DELETE} /*delete[](void*, nothrow)*/
  };
  // clang-format on

 public:
  void clear();
  void visitModuleGlobals(llvm::Module& m);
  void visitCallBase(llvm::CallBase& cb);
  void visitMallocLike(llvm::CallBase& ci, MemOpKind k);
  void visitFreeLike(llvm::CallBase& ci, MemOpKind k);
  void visitAllocaInst(llvm::AllocaInst& ai);
};

}  // namespace finder
}  // namespace typeart

#endif /* LIB_MEMOPVISITOR_H_ */

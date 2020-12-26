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
  //clang-format off
  const llvm::StringMap<MemOpKind> alloc_map{
      {"malloc", MemOpKind::MallocLike},
      {"calloc", MemOpKind::CallocLike},
      {"realloc", MemOpKind::ReallocLike},
      {"aligned_alloc", MemOpKind::AlignedAllocLike},
      {"_Znwm", MemOpKind::NewLike},                                 /*new(unsigned long)*/
      {"_Znwj", MemOpKind::NewLike},                                 /*new(unsigned int)*/
      {"_Znam", MemOpKind::NewLike},                                 /*new[](unsigned long)*/
      {"_Znaj", MemOpKind::NewLike},                                 /*new[](unsigned int)*/
      {"_ZnwjRKSt9nothrow_t", MemOpKind::MallocLike},                /*new(unsigned int, nothrow)*/
      {"_ZnwmRKSt9nothrow_t", MemOpKind::MallocLike},                /*new(unsigned long, nothrow)*/
      {"_ZnajRKSt9nothrow_t", MemOpKind::MallocLike},                /*new[](unsigned int, nothrow)*/
      {"_ZnamRKSt9nothrow_t", MemOpKind::MallocLike},                /*new[](unsigned long, nothrow)*/
      {"_ZnwjSt11align_val_t", MemOpKind::NewLike},                  /*new(unsigned int, align_val_t)*/
      {"_ZnwjSt11align_val_tRKSt9nothrow_t", MemOpKind::MallocLike}, /*new(unsigned int, align_val_t, nothrow)*/
      {"_ZnwmSt11align_val_t", MemOpKind::NewLike},                  /*new(unsigned long, align_val_t)*/
      {"_ZnwmSt11align_val_tRKSt9nothrow_t", MemOpKind::MallocLike}, /*new(unsigned long, align_val_t, nothrow)*/
      {"_ZnajSt11align_val_t", MemOpKind::NewLike},                  /*new[](unsigned int, align_val_t)*/
      {"_ZnajSt11align_val_tRKSt9nothrow_t", MemOpKind::MallocLike}, /*new[](unsigned int, align_val_t, nothrow)*/
      {"_ZnamSt11align_val_t", MemOpKind::NewLike},                  /*new[](unsigned long, align_val_t)*/
      {"_ZnamSt11align_val_tRKSt9nothrow_t", MemOpKind::MallocLike}, /*new[](unsigned long, align_val_t, nothrow)*/
  };

  const llvm::StringMap<MemOpKind> dealloc_map{
      {"free", MemOpKind::FreeLike},
      {"_ZdlPv", MemOpKind::DeleteLike},                              /*delete(void*)*/
      {"_ZdaPv", MemOpKind::DeleteLike},                              /*delete[](void*)*/
      {"_ZdlPvj", MemOpKind::DeleteLike},                             /*delete(void*, uint)*/
      {"_ZdlPvm", MemOpKind::DeleteLike},                             /*delete(void*, ulong)*/
      {"_ZdlPvRKSt9nothrow_t", MemOpKind::DeleteLike},                /*delete(void*, nothrow)*/
      {"_ZdaPvj", MemOpKind::DeleteLike},                             /*delete[](void*, uint)*/
      {"_ZdaPvm", MemOpKind::DeleteLike},                             /*delete[](void*, ulong)*/
      {"_ZdaPvRKSt9nothrow_t", MemOpKind::DeleteLike},                /*delete[](void*, nothrow)*/
      {"_ZdaPvSt11align_val_tRKSt9nothrow_t", MemOpKind::DeleteLike}, /* delete(void*, align_val_t, nothrow) */
      {"_ZdlPvSt11align_val_tRKSt9nothrow_t", MemOpKind::DeleteLike}, /* delete[](void*, align_val_t, nothrow) */
      {"_ZdlPvjSt11align_val_t", MemOpKind::DeleteLike},              /* delete(void*, unsigned long, align_val_t) */
      {"_ZdlPvmSt11align_val_t", MemOpKind::DeleteLike},              /* delete(void*, unsigned long, align_val_t) */
      {"_ZdaPvjSt11align_val_t", MemOpKind::DeleteLike},              /* delete[](void*, unsigned int, align_val_t) */
      {"_ZdaPvmSt11align_val_t", MemOpKind::DeleteLike},              /* delete[](void*, unsigned long, align_val_t) */
  };
  //clang-format off

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

//
// Created by ahueck on 08.10.20.
//

#ifndef TYPEART_MEMOPDATA_H
#define TYPEART_MEMOPDATA_H

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

namespace llvm {
class CallBase;
class BitCastInst;
class AllocaInst;
class GlobalVariable;
}  // namespace llvm

namespace typeart {
enum class MemOpKind : uint8_t {
  NewLike            = 1 << 0,            // allocates, never null
  MallocLike         = 1 << 1 | NewLike,  // allocates, maybe null
  AlignedAllocLike   = 1 << 2,            // allocates aligned, maybe null
  CallocLike         = 1 << 3,            // allocates zeroed
  ReallocLike        = 1 << 4,            // re-allocated (existing) memory
  FreeLike           = 1 << 5,            // free memory
  DeleteLike         = 1 << 6,            // delete (cpp) memory
  MallocOrCallocLike = MallocLike | CallocLike | AlignedAllocLike,
  AllocLike          = MallocOrCallocLike,
  AnyAlloc           = AllocLike | ReallocLike,
  AnyFree            = FreeLike | DeleteLike
};

// TyCart - BEGIN
enum class AssertKind { TYPE, TYPELEN, TYCART, TYCART_FTI_T, TYCART_AUTO, INTROSPECT_MARK };
// TyCart - END

struct MemOps {
  inline llvm::Optional<MemOpKind> kind(llvm::StringRef function) const {
    if (auto alloc = allocKind(function)) {
      return alloc;
    }
    if (auto dealloc = deallocKind(function)) {
      return dealloc;
    }
    return llvm::None;
  }

  inline llvm::Optional<MemOpKind> allocKind(llvm::StringRef function) const {
    if (auto it = alloc_map.find(function); it != std::end(alloc_map)) {
      return {(*it).second};
    }
    return llvm::None;
  }

  inline llvm::Optional<MemOpKind> deallocKind(llvm::StringRef function) const {
    if (auto it = dealloc_map.find(function); it != std::end(dealloc_map)) {
      return {(*it).second};
    }
    return llvm::None;
  }

  const llvm::StringMap<MemOpKind>& allocs() const {
    return alloc_map;
  }

  const llvm::StringMap<MemOpKind>& deallocs() const {
    return dealloc_map;
  }

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
  
  // TyCart - BEGIN
  const llvm::StringMap<AssertKind> assert_map{
      {"__typeart_assert_type_stub", AssertKind::TYPE},
      {"__typeart_assert_type_stub_len", AssertKind::TYPELEN},
      {"__tycart_assert_stub", AssertKind::TYCART},
      {"__tycart_register_FTI_t_stub", AssertKind::TYCART_FTI_T},
      {"__tycart_assert_auto_stub", AssertKind::TYCART_AUTO}
  };
  // TyCart - END
  
  //clang-format off
};

struct MallocData {
  llvm::CallBase* call{nullptr};
  llvm::BitCastInst* primary{nullptr};  // Non-null if non (void*) cast exists
  llvm::SmallPtrSet<llvm::BitCastInst*, 4> bitcasts;
  MemOpKind kind;
  bool is_invoke{false};
};

struct FreeData {
  llvm::CallBase* call{nullptr};
  MemOpKind kind;
  bool is_invoke{false};
};

struct AllocaData {
  llvm::AllocaInst* alloca{nullptr};
  size_t array_size;
  bool is_vla{false};
};

// TyCart - BEGIN
struct AssertData {
  llvm::CallBase* call{nullptr};
  AssertKind kind;
};
// TyCart - END

struct GlobalData {
  llvm::GlobalVariable* global{nullptr};
};

using GlobalDataList = llvm::SmallVector<GlobalData, 8>;
using MallocDataList = llvm::SmallVector<MallocData, 8>;
using FreeDataList   = llvm::SmallVector<FreeData, 8>;
using AllocaDataList = llvm::SmallVector<AllocaData, 8>;

// TyCart - BEGIN
using AssertDataList = llvm::SmallVector<AssertData, 8>;
// TyCart - END

}  // namespace typeart
#endif  // TYPEART_MEMOPDATA_H

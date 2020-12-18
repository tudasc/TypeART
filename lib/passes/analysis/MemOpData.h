//
// Created by ahueck on 08.10.20.
//

#ifndef TYPEART_MEMOPDATA_H
#define TYPEART_MEMOPDATA_H

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

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

struct GlobalData {
  llvm::GlobalVariable* global{nullptr};
};

using GlobalDataList = llvm::SmallVector<GlobalData, 8>;
using MallocDataList = llvm::SmallVector<MallocData, 8>;
using FreeDataList   = llvm::SmallVector<FreeData, 8>;
using AllocaDataList = llvm::SmallVector<AllocaData, 8>;

}  // namespace typeart
#endif  // TYPEART_MEMOPDATA_H

//
// Created by ahueck on 08.10.20.
//

#ifndef TYPEART_MEMOPDATA_H
#define TYPEART_MEMOPDATA_H

#include "llvm/ADT/SmallPtrSet.h"

namespace llvm {
class CallBase;
class BitCastInst;
class AllocaInst;
class GlobalVariable;
}  // namespace llvm

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

}  // namespace typeart
#endif  // TYPEART_MEMOPDATA_H

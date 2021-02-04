//
// Created by sebastian on 11.01.21.
//

#ifndef TYPEART_ALLOCATIONTRACKING_H
#define TYPEART_ALLOCATIONTRACKING_H

#include "AccessCounter.h"
#include "RuntimeData.h"

#include <shared_mutex>

namespace llvm {
template <typename T>
class Optional;
}  // namespace llvm

namespace typeart {

class TypeDB;

enum class AllocState : unsigned {
  NO_INIT      = 1 << 0,
  OK           = 1 << 1,
  ADDR_SKIPPED = 1 << 2,
  NULL_PTR     = 1 << 3,
  ZERO_COUNT   = 1 << 4,
  NULL_ZERO    = 1 << 5,
  ADDR_REUSE   = 1 << 6,
  UNKNOWN_ID   = 1 << 7
};

enum class FreeState : unsigned {
  NO_INIT      = 1 << 0,
  OK           = 1 << 1,
  ADDR_SKIPPED = 1 << 2,
  NULL_PTR     = 1 << 3,
  UNREG_ADDR   = 1 << 4,
};

class AllocationTracker {
  RuntimeT::PointerMapSafe allocTypesSafe;
  const TypeDB& typeDB;
  Recorder& recorder;

 public:
  AllocationTracker(const TypeDB& db, Recorder& recorder);

  void onAlloc(const void* addr, int typeID, size_t count, const void* retAddr);

  void onAllocStack(const void* addr, int typeID, size_t count, const void* retAddr);

  void onAllocGlobal(const void* addr, int typeID, size_t count, const void* retAddr);

  void onFreeHeap(const void* addr, const void* retAddr);

  void onLeaveScope(int alloca_count, const void* retAddr);

  llvm::Optional<RuntimeT::MapEntry> findBaseAlloc(const void* addr);

 private:
  AllocState doAlloc(const void* addr, int typeID, size_t count, const void* retAddr);

  FreeState doFreeHeap(const void* addr, const void* retAddr);
};

}  // namespace typeart

#endif  // TYPEART_ALLOCATIONTRACKING_H

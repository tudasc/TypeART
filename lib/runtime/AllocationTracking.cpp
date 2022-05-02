// TypeART library
//
// Copyright (c) 2017-2022 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#include "AllocationTracking.h"

#include "AccessCounter.h"
#include "CallbackInterface.h"
#include "Runtime.h"
#include "RuntimeData.h"
#include "TypeDB.h"
#include "support/Logger.h"

#include "llvm/ADT/Optional.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <iterator>
#include <type_traits>
#include <vector>

#ifdef TYPEART_BTREE
using namespace btree;
#endif

#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#define CONCAT_(x, y) x##y
#define CONCAT(x, y)  CONCAT_(x, y)
#define GUARDNAME     CONCAT(typeart_guard_, __LINE__)

#define TYPEART_RUNTIME_GUARD     \
  typeart::RTGuard GUARDNAME;     \
  if (!GUARDNAME.shouldTrack()) { \
    return;                       \
  }

namespace typeart {

namespace detail {
template <class...>
constexpr std::false_type always_false{};
}  // namespace detail

template <typename Enum>
inline Enum operator|(Enum lhs, Enum rhs) {
  if constexpr (std::is_enum_v<Enum> && (std::is_same_v<Enum, AllocState> || std::is_same_v<Enum, FreeState>)) {
    using enum_t = typename std::underlying_type<Enum>::type;
    return static_cast<Enum>(static_cast<enum_t>(lhs) | static_cast<enum_t>(rhs));
  } else {
    static_assert(detail::always_false<Enum>);
  }
}
template <typename Enum>
inline void operator|=(Enum& lhs, Enum rhs) {
  if constexpr (std::is_enum_v<Enum> && (std::is_same_v<Enum, AllocState> || std::is_same_v<Enum, FreeState>)) {
    lhs = lhs | rhs;
  } else {
    static_assert(detail::always_false<Enum>);
  }
}

template <typename Enum>
inline Enum operator&(Enum lhs, Enum rhs) {
  if constexpr (std::is_enum_v<Enum> && std::is_same_v<Enum, AllocState>) {
    using enum_t = typename std::underlying_type<Enum>::type;
    return static_cast<Enum>(static_cast<enum_t>(lhs) & static_cast<enum_t>(rhs));
  } else {
    static_assert(detail::always_false<Enum>);
  }
}

template <typename Enum>
inline typename std::underlying_type<Enum>::type operator==(Enum lhs, Enum rhs) {
  if constexpr (std::is_enum_v<Enum> && std::is_same_v<Enum, AllocState>) {
    using enum_t = typename std::underlying_type<Enum>::type;
    return static_cast<enum_t>(lhs) & static_cast<enum_t>(rhs);
  } else {
    static_assert(detail::always_false<Enum>);
  }
}

using namespace debug;

namespace {
struct ThreadData final {
  RuntimeT::Stack stackVars;

  ThreadData() {
    stackVars.reserve(RuntimeT::StackReserve);
  }
};

thread_local ThreadData threadData;

}  // namespace

AllocationTracker::AllocationTracker(const TypeDB& db, Recorder& recorder) : typeDB{db}, recorder{recorder} {
}

void AllocationTracker::onAlloc(const void* addr, int typeId, size_t count, const void* retAddr) {
  const auto status = doAlloc(addr, typeId, count, retAddr);
  if (status != AllocState::ADDR_SKIPPED) {
    recorder.incHeapAlloc(typeId, count);
  }
  LOG_TRACE("Alloc " << toString(addr, typeId, count, retAddr) << " " << 'H');
}

void AllocationTracker::onAllocStack(const void* addr, int typeId, size_t count, const void* retAddr) {
  const auto status = doAlloc(addr, typeId, count, retAddr);
  if (status != AllocState::ADDR_SKIPPED) {
    threadData.stackVars.push_back(addr);
    recorder.incStackAlloc(typeId, count);
  }
  LOG_TRACE("Alloc " << toString(addr, typeId, count, retAddr) << " " << 'S');
}

void AllocationTracker::onAllocGlobal(const void* addr, int typeId, size_t count, const void* retAddr) {
  const auto status = doAlloc(addr, typeId, count, retAddr);
  if (status != AllocState::ADDR_SKIPPED) {
    recorder.incGlobalAlloc(typeId, count);
  }
  LOG_TRACE("Alloc " << toString(addr, typeId, count, retAddr) << " " << 'G');
}

AllocState AllocationTracker::doAlloc(const void* addr, int typeId, size_t count, const void* retAddr) {
  AllocState status = AllocState::NO_INIT;
  if (unlikely(!typeDB.isValid(typeId))) {
    status |= AllocState::UNKNOWN_ID;
    LOG_ERROR("Allocation of unknown type " << toString(addr, typeId, count, retAddr));
  }

  // Calling malloc with size 0 may return a nullptr or some address that can not be written to.
  // In the second case, the allocation is tracked anyway so that onFree() does not report an error.
  // On the other hand, an allocation on address 0x0 with size > 0 is an actual error.
  if (unlikely(count == 0)) {
    recorder.incZeroLengthAddr();
    status |= AllocState::ZERO_COUNT;
    LOG_WARNING("Zero-size allocation " << toString(addr, typeId, count, retAddr));
    if (addr == nullptr) {
      recorder.incZeroLengthAndNullAddr();
      LOG_ERROR("Zero-size and nullptr allocation " << toString(addr, typeId, count, retAddr));
      return status | AllocState::NULL_ZERO | AllocState::ADDR_SKIPPED;
    }
  } else if (unlikely(addr == nullptr)) {
    recorder.incNullAddr();
    LOG_ERROR("Nullptr allocation " << toString(addr, typeId, count, retAddr));
    return status | AllocState::NULL_PTR | AllocState::ADDR_SKIPPED;
  }

  const auto overridden = wrapper.put(addr, PointerInfo{typeId, count, retAddr});

  if (unlikely(overridden)) {
    recorder.incAddrReuse();
    status |= AllocState::ADDR_REUSE;
    LOG_WARNING("Pointer already in map " << toString(addr, typeId, count, retAddr));
    // LOG_WARNING("Overridden data in map " << toString(addr, def));
  }

  return status | AllocState::OK;
}

FreeState AllocationTracker::doFreeHeap(const void* addr, const void* retAddr) {
  if (unlikely(addr == nullptr)) {
    LOG_ERROR("Free on nullptr "
              << "(" << retAddr << ")");
    return FreeState::ADDR_SKIPPED | FreeState::NULL_PTR;
  }

  llvm::Optional<PointerInfo> removed = wrapper.remove(addr);

  if (unlikely(!removed)) {
    LOG_ERROR("Free on unregistered address " << addr << " (" << retAddr << ")");
    return FreeState::ADDR_SKIPPED | FreeState::UNREG_ADDR;
  }

  LOG_TRACE("Free " << toString(addr, *removed));
  if constexpr (!std::is_same_v<Recorder, softcounter::NoneRecorder>) {
    recorder.incHeapFree(removed->typeId, removed->count);
  }
  return FreeState::OK;
}

void AllocationTracker::onFreeHeap(const void* addr, const void* retAddr) {
  const auto status = doFreeHeap(addr, retAddr);
  if (FreeState::OK == status) {
    recorder.decHeapAlloc();
  }
}

void AllocationTracker::onLeaveScope(int alloca_count, const void* retAddr) {
  if (unlikely(alloca_count > static_cast<int>(threadData.stackVars.size()))) {
    LOG_ERROR("Stack is smaller than requested de-allocation count. alloca_count: " << alloca_count << ". size: "
                                                                                    << threadData.stackVars.size());
    alloca_count = threadData.stackVars.size();
  }

  const auto cend      = threadData.stackVars.cend();
  const auto start_pos = (cend - alloca_count);
  LOG_TRACE("Freeing stack (" << alloca_count << ")  " << std::distance(start_pos, threadData.stackVars.cend()))

  wrapper.remove_range(start_pos, cend, [&](llvm::Optional<PointerInfo>& removed, MemAddr addr) {
    if (unlikely(!removed)) {
      LOG_ERROR("Free on unregistered address " << addr << " (" << retAddr << ")");
    } else {
      LOG_TRACE("Free " << toString(addr, *removed));
      if constexpr (!std::is_same_v<Recorder, softcounter::NoneRecorder>) {
        recorder.incStackFree(removed->typeId, removed->count);
      }
    }
  });

  threadData.stackVars.erase(start_pos, cend);
  recorder.decStackAlloc(alloca_count);
  LOG_TRACE("Stack after free: " << threadData.stackVars.size());
}
// Base address
llvm::Optional<RuntimeT::MapEntry> AllocationTracker::findBaseAlloc(const void* addr) {
  return wrapper.find(addr);
}

}  // namespace typeart

void __typeart_alloc(const void* addr, int typeId, size_t count) {
  TYPEART_RUNTIME_GUARD;
  const void* retAddr = __builtin_return_address(0);
  typeart::RuntimeSystem::get().allocTracker.onAlloc(addr, typeId, count, retAddr);
}

void __typeart_alloc_stack(const void* addr, int typeId, size_t count) {
  TYPEART_RUNTIME_GUARD;
  const void* retAddr = __builtin_return_address(0);
  typeart::RuntimeSystem::get().allocTracker.onAllocStack(addr, typeId, count, retAddr);
}

void __typeart_alloc_global(const void* addr, int typeId, size_t count) {
  TYPEART_RUNTIME_GUARD;
  const void* retAddr = __builtin_return_address(0);
  typeart::RuntimeSystem::get().allocTracker.onAllocGlobal(addr, typeId, count, retAddr);
}

void __typeart_free(const void* addr) {
  TYPEART_RUNTIME_GUARD;
  const void* retAddr = __builtin_return_address(0);
  typeart::RuntimeSystem::get().allocTracker.onFreeHeap(addr, retAddr);
}

void __typeart_leave_scope(int alloca_count) {
  TYPEART_RUNTIME_GUARD;
  const void* retAddr = __builtin_return_address(0);
  typeart::RuntimeSystem::get().allocTracker.onLeaveScope(alloca_count, retAddr);
}

void __typeart_alloc_omp(const void* addr, int typeId, size_t count) {
  TYPEART_RUNTIME_GUARD;
  const void* retAddr = __builtin_return_address(0);
  typeart::RuntimeSystem::get().allocTracker.onAlloc(addr, typeId, count, retAddr);
  typeart::RuntimeSystem::get().recorder.incOmpContextHeap();
}

void __typeart_alloc_stack_omp(const void* addr, int typeId, size_t count) {
  TYPEART_RUNTIME_GUARD;
  const void* retAddr = __builtin_return_address(0);
  typeart::RuntimeSystem::get().allocTracker.onAllocStack(addr, typeId, count, retAddr);
  typeart::RuntimeSystem::get().recorder.incOmpContextStack();
}

void __typeart_free_omp(const void* addr) {
  TYPEART_RUNTIME_GUARD;
  const void* retAddr = __builtin_return_address(0);
  typeart::RuntimeSystem::get().allocTracker.onFreeHeap(addr, retAddr);
  typeart::RuntimeSystem::get().recorder.incOmpContextFree();
}

void __typeart_leave_scope_omp(int alloca_count) {
  TYPEART_RUNTIME_GUARD;
  const void* retAddr = __builtin_return_address(0);
  typeart::RuntimeSystem::get().allocTracker.onLeaveScope(alloca_count, retAddr);
}

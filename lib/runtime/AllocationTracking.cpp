//
// Created by sebastian on 07.01.21.
//

#include "AllocationTracking.h"

#include "AccessCounter.h"
#include "CallbackInterface.h"
#include "Runtime.h"
#include "RuntimeData.h"
#include "TypeDB.h"
#include "support/Logger.h"

#include <algorithm>
#include <string>

#ifdef USE_BTREE
using namespace btree;
#endif

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#define RUNTIME_GUARD_BEGIN        \
  if (typeart::threadData.rtScope) { \
    return;                        \
  }                                \
  typeart::threadData.rtScope = true
#define RUNTIME_GUARD_END typeart::threadData.rtScope = false

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
  struct ThreadData {
    bool rtScope{false};
    RuntimeT::Stack stackVars;

    ThreadData() {
      stackVars.reserve(RuntimeT::StackReserve);
    }
  };

  thread_local ThreadData threadData;

}

AllocationTracker::AllocationTracker(const TypeDB& db, Recorder& recorder) : typeDB{db}, recorder{recorder} {
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
    LOG_ERROR("Zero-size allocation " << toString(addr, typeId, count, retAddr));
    return status | AllocState::NULL_PTR | AllocState::ADDR_SKIPPED;
  }

  auto& def = allocTypes[addr];

  if (unlikely(def.typeId != -1)) {
    recorder.incAddrReuse();
    status |= AllocState::ADDR_REUSE;
    LOG_WARNING("Pointer already in map " << toString(addr, typeId, count, retAddr));
    LOG_WARNING("Overriden data in map " << toString(addr, def));
  }

  def.typeId = typeId;
  def.count  = count;
  def.debug  = retAddr;

  return status | AllocState::OK;
}

void AllocationTracker::onAlloc(const void* addr, int typeId, size_t count, const void* retAddr) {
  std::lock_guard<std::shared_mutex> guard(allocMutex);
  const auto status = doAlloc(addr, typeId, count, retAddr);
  if (status != AllocState::ADDR_SKIPPED) {
    recorder.incHeapAlloc(typeId, count);
  }
  LOG_TRACE("Alloc " << toString(addr, typeId, count, retAddr) << " " << 'H');
}

void AllocationTracker::onAllocStack(const void* addr, int typeId, size_t count, const void* retAddr) {
  std::lock_guard<std::shared_mutex> guard(allocMutex);
  const auto status = doAlloc(addr, typeId, count, retAddr);
  if (status != AllocState::ADDR_SKIPPED) {
    threadData.stackVars.push_back(addr);
    recorder.incStackAlloc(typeId, count);
  }
  LOG_TRACE("Alloc " << toString(addr, typeId, count, retAddr) << " " << 'S');
}

void AllocationTracker::onAllocGlobal(const void* addr, int typeId, size_t count, const void* retAddr) {
  std::lock_guard<std::shared_mutex> guard(allocMutex);
  const auto status = doAlloc(addr, typeId, count, retAddr);
  if (status != AllocState::ADDR_SKIPPED) {
    recorder.incGlobalAlloc(typeId, count);
  }
  LOG_TRACE("Alloc " << toString(addr, typeId, count, retAddr) << " " << 'G');
}

template <bool stack>
FreeState AllocationTracker::doFree(const void* addr, const void* retAddr) {
  if (unlikely(addr == nullptr)) {
    LOG_ERROR("Free on nullptr "
              << "(" << retAddr << ")");
    return FreeState::ADDR_SKIPPED | FreeState::NULL_PTR;
  }

  const auto it = allocTypes.find(addr);

  if (likely(it != allocTypes.end())) {
    LOG_TRACE("Free " << toString((*it).first, (*it).second));

    if constexpr (!std::is_same_v<Recorder, softcounter::NoneRecorder>) {
      const auto typeId = it->second.typeId;
      const auto count  = it->second.count;
      if (stack) {
        recorder.incStackFree(typeId, count);
      } else {
        recorder.incHeapFree(typeId, count);
      }
    }

    allocTypes.erase(it);
  } else {
    LOG_ERROR("Free on unregistered address " << addr << " (" << retAddr << ")");
    return FreeState::ADDR_SKIPPED | FreeState::UNREG_ADDR;
  }

  return FreeState::OK;
}

void AllocationTracker::onFreeHeap(const void* addr, const void* retAddr) {
  std::lock_guard<std::shared_mutex> guard(allocMutex);
  const auto status = doFree<false>(addr, retAddr);
  if (FreeState::OK == status) {
    recorder.decHeapAlloc();
  }
}

void AllocationTracker::onLeaveScope(int alloca_count, const void* retAddr) {
  std::lock_guard<std::shared_mutex> guard(allocMutex);
  if (unlikely(alloca_count > threadData.stackVars.size())) {
    LOG_ERROR("Stack is smaller than requested de-allocation count. alloca_count: " << alloca_count
                                                                                    << ". size: " << threadData.stackVars.size());
    alloca_count = threadData.stackVars.size();
  }

  const auto cend      = threadData.stackVars.cend();
  const auto start_pos = (cend - alloca_count);
  LOG_TRACE("Freeing stack (" << alloca_count << ")  " << std::distance(start_pos, threadData.stackVars.cend()))
  std::for_each(start_pos, cend, [this, &retAddr](const void* addr) { doFree<true>(addr, retAddr); });
  threadData.stackVars.erase(start_pos, cend);
  recorder.decStackAlloc(alloca_count);
  LOG_TRACE("Stack after free: " << threadData.stackVars.size());
}

llvm::Optional<RuntimeT::MapEntry> AllocationTracker::findBaseAlloc(const void* addr) {
  std::shared_lock guard(allocMutex);
  if (allocTypes.empty() || addr < allocTypes.begin()->first) {
    return llvm::None;
  }

  auto it = allocTypes.lower_bound(addr);
  if (it == allocTypes.end()) {
    // No element bigger than base address
    return {*allocTypes.rbegin()};
  }

  if (it->first == addr) {
    // Exact match
    return {*it};
  }
  // Base address
  return {*std::prev(it)};
}

}  // namespace typeart

void __typeart_alloc(const void* addr, int typeId, size_t count) {
  RUNTIME_GUARD_BEGIN;
  const void* retAddr = __builtin_return_address(0);
  typeart::kRuntimeSystem.allocTracker.onAlloc(addr, typeId, count, retAddr);
  RUNTIME_GUARD_END;
}

void __typeart_alloc_stack(const void* addr, int typeId, size_t count) {
  RUNTIME_GUARD_BEGIN;
  const void* retAddr = __builtin_return_address(0);
  typeart::kRuntimeSystem.allocTracker.onAllocStack(addr, typeId, count, retAddr);
  RUNTIME_GUARD_END;
}

void __typeart_alloc_global(const void* addr, int typeId, size_t count) {
  RUNTIME_GUARD_BEGIN;
  const void* retAddr = __builtin_return_address(0);
  typeart::kRuntimeSystem.allocTracker.onAllocGlobal(addr, typeId, count, retAddr);
  RUNTIME_GUARD_END;
}

void __typeart_free(const void* addr) {
  RUNTIME_GUARD_BEGIN;
  const void* retAddr = __builtin_return_address(0);
  typeart::kRuntimeSystem.allocTracker.onFreeHeap(addr, retAddr);
  RUNTIME_GUARD_END;
}

void __typeart_leave_scope(int alloca_count) {
  RUNTIME_GUARD_BEGIN;
  const void* retAddr = __builtin_return_address(0);
  typeart::kRuntimeSystem.allocTracker.onLeaveScope(alloca_count, retAddr);
  RUNTIME_GUARD_END;
}

//
// Created by sebastian on 11.01.21.
//

#ifndef TYPEART_ALLOCATIONTRACKING_H
#define TYPEART_ALLOCATIONTRACKING_H

#include "AccessCounter.h"
#include "RuntimeData.h"
#include "TypeResolution.h"

namespace llvm {
template <typename T>
class Optional;
}  // namespace llvm

namespace typeart {

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

class AllocationTracker {
  RuntimeT::PointerMap allocTypes;
  RuntimeT::Stack stackVars;

 public:
  AllocationTracker() {
    stackVars.reserve(RuntimeT::StackReserve);
  }

  void onAlloc(const void* addr, int typeID, size_t count, const void* retAddr);

  void onAllocStack(const void* addr, int typeID, size_t count, const void* retAddr);

  void onAllocGlobal(const void* addr, int typeID, size_t count, const void* retAddr);

  void onFreeHeap(const void* addr, const void* retAddr);

  void onLeaveScope(int alloca_count, const void* retAddr);

  llvm::Optional<RuntimeT::MapEntry> findBaseAlloc(const void* addr);

 private:
  AllocState doAlloc(const void* addr, int typeID, size_t count, const void* retAddr);

  template <bool stack>
  FreeState doFree(const void* addr, const void* retAddr);
};

}  // namespace typeart

#endif  // TYPEART_ALLOCATIONTRACKING_H

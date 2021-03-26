//
// Created by ahueck on 26.03.21.
//

#ifndef TYPEART_ALLOCMAPWRAPPER_H
#define TYPEART_ALLOCMAPWRAPPER_H

#include "RuntimeData.h"

#include "llvm/ADT/Optional.h"

#include <algorithm>

namespace typeart {
enum class Mode { thread_safe = 0, thread_unsafe };

namespace mixin {

namespace detail {
template <typename Map>
struct PtrWrap final {
  Map& m;
  Map* operator->() {
    return &m;
  }
  Map& operator*() {
    return m;
  }
  const Map* operator->() const {
    return &m;
  }
  const Map& operator*() const {
    return m;
  }
};

template <typename Map>
[[nodiscard]] PtrWrap<Map> as_ptr(Map& m) {
  return PtrWrap<Map>{m};
}
}  // namespace detail

struct MapOp {
  template <typename Xlocked>
  [[nodiscard]] inline static bool do_put(Xlocked&& lockedAllocs, MemAddr addr, const RuntimeT::MappedType& data) {
    auto& def             = (*lockedAllocs)[addr];
    const bool overridden = (def.typeId != -1);
    def                   = data;
    return overridden;
  }

  template <typename Slocked>
  [[nodiscard]] inline static llvm::Optional<RuntimeT::MapEntry> do_find(Slocked&& slockedAllocs, MemAddr addr) {
    if (slockedAllocs->empty() || addr < slockedAllocs->begin()->first) {
      return llvm::None;
    }

    auto it = slockedAllocs->lower_bound(addr);
    if (it == slockedAllocs->end()) {
      // No element bigger than base address
      return {*slockedAllocs->rbegin()};
    }

    if (it->first == addr) {
      // Exact match
      return {*it};
    }
    // Base address
    return {*std::prev(it)};
  }

  template <typename Xlocked>
  [[nodiscard]] inline static llvm::Optional<RuntimeT::MappedType> do_remove(Xlocked&& allocs, MemAddr addr) {
    const auto it = allocs->find(addr);
    if (it != allocs->end()) {
      auto removed = it->second;
      allocs->erase(it);
      return removed;
    }
    return llvm::None;
  }
};

template <typename BaseOp>
struct StandardMap : public BaseOp {
  RuntimeT::PointerMapBaseT allocTypesSafe;
  mutable std::shared_mutex alloc_m;

  template <Mode m = Mode::thread_safe>
  [[nodiscard]] inline llvm::Optional<RuntimeT::MapEntry> find(MemAddr addr) const {
    using namespace detail;
    if constexpr (m == Mode::thread_safe) {
      std::shared_lock<std::shared_mutex> guard(alloc_m);
      return BaseOp::do_find(as_ptr(allocTypesSafe), addr);
    } else {
      return BaseOp::do_find(as_ptr(allocTypesSafe), addr);
    }
  }

  template <Mode m = Mode::thread_safe>
  [[nodiscard]] inline bool put(MemAddr addr, const RuntimeT::MappedType& entry) {
    using namespace detail;
    if constexpr (m == Mode::thread_safe) {
      std::lock_guard<std::shared_mutex> guard(alloc_m);
      return BaseOp::do_put(as_ptr(allocTypesSafe), addr, entry);
    } else {
      return BaseOp::do_put(as_ptr(allocTypesSafe), addr, entry);
    }
  }

  template <Mode m = Mode::thread_safe>
  [[nodiscard]] inline llvm::Optional<RuntimeT::MappedType> remove(MemAddr addr) {
    using namespace detail;
    if constexpr (m == Mode::thread_safe) {
      std::lock_guard<std::shared_mutex> guard(alloc_m);
      return BaseOp::do_remove(as_ptr(allocTypesSafe), addr);
    } else {
      return BaseOp::do_remove(as_ptr(allocTypesSafe), addr);
    }
  }
};

#ifdef USE_SAFEPTR
template <typename BaseOp>
struct SafePtrdMap : public BaseOp {
  RuntimeT::PointerMap allocTypesSafe;

  template <Mode m = Mode::thread_safe>
  [[nodiscard]] inline llvm::Optional<RuntimeT::MapEntry> find(MemAddr addr) const {
    static_assert(m != Mode::thread_unsafe, "SafePtrMap is always thread safe.");
    auto slockedAllocs = sf::slock_safe_ptr(allocTypesSafe);
    return BaseOp::do_find(slockedAllocs, addr);
  }
  template <Mode m = Mode::thread_safe>
  [[nodiscard]] inline bool put(MemAddr addr, const RuntimeT::MappedType& entry) {
    static_assert(m != Mode::thread_unsafe, "SafePtrMap is always thread safe.");
    auto guard = sf::xlock_safe_ptr(allocTypesSafe);
    return BaseOp::do_put(guard, addr, entry);
  }
  template <Mode m = Mode::thread_safe>
  [[nodiscard]] inline llvm::Optional<RuntimeT::MappedType> remove(MemAddr addr) {
    static_assert(m != Mode::thread_unsafe, "SafePtrMap is always thread safe.");
    auto guard = sf::xlock_safe_ptr(allocTypesSafe);
    return BaseOp::do_remove(guard, addr);
  }
};
#endif
}  // namespace mixin

#ifdef USE_SAFEPTR
using PointerMap = mixin::SafePtrdMap<mixin::MapOp>;
#else
using PointerMap = mixin::StandardMap<mixin::MapOp>;
;
#endif

}  // namespace typeart

#endif  // TYPEART_ALLOCMAPWRAPPER_H

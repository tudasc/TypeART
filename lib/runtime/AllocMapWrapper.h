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
template <typename Derived>
class Wrapper {
 public:
  Derived& get() {
    return *static_cast<Derived*>(this);
  }

  template <Mode M = Mode::thread_safe>
  [[nodiscard]] inline llvm::Optional<RuntimeT::MappedType> find(MemAddr addr) {
    return get().template find<M>(addr);
  }

  template <Mode m = Mode::thread_safe>
  [[nodiscard]] inline bool put(MemAddr addr, const RuntimeT::MappedType& entry) {
    return get().template put<m>(entry);
  }

  template <Mode m = Mode::thread_safe>
  [[nodiscard]] inline bool remove(MemAddr addr) {
    return get().template remove<m>(addr);
  }

 protected:
  template <typename Xlocked>
  [[nodiscard]] inline bool do_put(Xlocked&& lockedAllocs, MemAddr addr, const RuntimeT::MappedType& data) {
    auto& def             = (*lockedAllocs)[addr];
    const bool overridden = (def.typeId != -1);
    def                   = data;
    return overridden;
  }

  template <typename Slocked>
  [[nodiscard]] inline llvm::Optional<RuntimeT::MapEntry> do_find(Slocked&& slockedAllocs, MemAddr addr) const {
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
  [[nodiscard]] inline llvm::Optional<RuntimeT::MappedType> do_remove(Xlocked&& allocs, MemAddr addr) {
    const auto it = allocs->find(addr);
    if (it != allocs->end()) {
      auto removed = it->second;
      allocs->erase(it);
      return removed;
    }
    return llvm::None;
  }
};

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

class MapWrapper : public Wrapper<MapWrapper> {
  RuntimeT::PointerMapBaseT allocTypesSafe;
  mutable std::shared_mutex alloc_m;

 public:
  template <Mode m = Mode::thread_safe>
  [[nodiscard]] inline llvm::Optional<RuntimeT::MapEntry> find(MemAddr addr) const {
    using namespace detail;
    if constexpr (m == Mode::thread_safe) {
      std::shared_lock<std::shared_mutex> guard(alloc_m);
      return do_find(as_ptr(allocTypesSafe), addr);
    } else {
      return do_find(as_ptr(allocTypesSafe), addr);
    }
  }

  template <Mode m = Mode::thread_safe>
  [[nodiscard]] inline bool put(MemAddr addr, const RuntimeT::MappedType& entry) {
    using namespace detail;
    if constexpr (m == Mode::thread_safe) {
      std::lock_guard<std::shared_mutex> guard(alloc_m);
      return do_put(as_ptr(allocTypesSafe), addr, entry);
    } else {
      return do_put(as_ptr(allocTypesSafe), addr, entry);
    }
  }

  template <Mode m = Mode::thread_safe>
  [[nodiscard]] inline llvm::Optional<RuntimeT::MappedType> remove(MemAddr addr) {
    using namespace detail;
    if constexpr (m == Mode::thread_safe) {
      std::lock_guard<std::shared_mutex> guard(alloc_m);
      return do_remove(as_ptr(allocTypesSafe), addr);
    } else {
      return do_remove(as_ptr(allocTypesSafe), addr);
    }
  }
};

#ifdef USE_SAFEPTR
class SafePtrWrapper : public Wrapper<SafePtrWrapper> {
  RuntimeT::PointerMap allocTypesSafe;

 public:
  template <Mode m = Mode::thread_safe>
  [[nodiscard]] inline llvm::Optional<RuntimeT::MapEntry> find(MemAddr addr) const {
    auto slockedAllocs = sf::slock_safe_ptr(allocTypesSafe);
    return do_find(slockedAllocs, addr);
  }
  template <Mode m = Mode::thread_safe>
  [[nodiscard]] inline bool put(MemAddr addr, const RuntimeT::MappedType& entry) {
    auto guard = sf::xlock_safe_ptr(allocTypesSafe);
    return do_put(guard, addr, entry);
  }
  template <Mode m = Mode::thread_safe>
  [[nodiscard]] inline llvm::Optional<RuntimeT::MappedType> remove(MemAddr addr) {
    auto guard = sf::xlock_safe_ptr(allocTypesSafe);
    return do_remove(guard, addr);
  }
};
#endif

#ifdef USE_SAFEPTR
using PointerMap = SafePtrWrapper;
#else
using PointerMap = MapWrapper;
#endif

}  // namespace typeart

#endif  // TYPEART_ALLOCMAPWRAPPER_H

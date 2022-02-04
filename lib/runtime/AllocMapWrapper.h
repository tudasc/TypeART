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

#ifndef TYPEART_ALLOCMAPWRAPPER_H
#define TYPEART_ALLOCMAPWRAPPER_H

#include "RuntimeData.h"

#include "llvm/ADT/Optional.h"

#include <algorithm>

namespace typeart {
namespace mixin {
enum class BulkOperation { remove = 0 };

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
[[nodiscard]] inline PtrWrap<Map> as_ptr(Map& m) {
  return PtrWrap<Map>{m};
}
}  // namespace detail

struct MapOp {
 private:
  RuntimeT::PointerMap map_;

 public:
  [[nodiscard]] const RuntimeT::PointerMap& map() const {
    return map_;
  }

  [[nodiscard]] RuntimeT::PointerMap& map() {
    return map_;
  }

  template <typename PointerMap>
  [[nodiscard]] inline static bool put(PointerMap&& xlocked_map, MemAddr addr, const RuntimeT::MappedType& data) {
    auto& def             = (*xlocked_map)[addr];
    const bool overridden = (def.typeId != -1);
    def                   = data;
    return overridden;
  }

  template <typename PointerMap>
  [[nodiscard]] inline static llvm::Optional<RuntimeT::MapEntry> find(PointerMap&& slocked_map, MemAddr addr) {
    if (slocked_map->empty() || addr < slocked_map->begin()->first) {
      return llvm::None;
    }

    auto it = slocked_map->lower_bound(addr);
    if (it == slocked_map->end()) {
      // No element bigger than base address
      return {*slocked_map->rbegin()};
    }

    if (it->first == addr) {
      // Exact match
      return {*it};
    }
    // Base address
    return {*std::prev(it)};
  }

  template <typename PointerMap>
  [[nodiscard]] inline static llvm::Optional<RuntimeT::MappedType> remove(PointerMap&& xlocked_map, MemAddr addr) {
    const auto it = xlocked_map->find(addr);
    if (it != xlocked_map->end()) {
      auto removed = it->second;
      xlocked_map->erase(it);
      return removed;
    }
    return llvm::None;
  }

  template <BulkOperation Operation, typename PointerMap, typename FwdIter, typename Callback>
  inline static void bulk_op(PointerMap&& xlocked_map, FwdIter&& s, FwdIter&& e, Callback&& log) {
    if constexpr (Operation == BulkOperation::remove) {
      std::for_each(s, e, [&xlocked_map, &log](MemAddr addr) {
        auto removed = remove(std::forward<PointerMap>(xlocked_map), addr);
        log(removed, addr);
      });
    } else {
      static_assert(true, "Unsupported operation");
    }
  }
};

template <typename BaseOp>
struct StandardMapBase : protected BaseOp {
  [[nodiscard]] inline llvm::Optional<RuntimeT::MapEntry> find(MemAddr addr) const {
    return BaseOp::find(detail::as_ptr(this->map()), addr);
  }

  [[nodiscard]] inline bool put(MemAddr addr, const RuntimeT::MappedType& entry) {
    return BaseOp::put(detail::as_ptr(this->map()), addr, entry);
  }

  [[nodiscard]] inline llvm::Optional<RuntimeT::MappedType> remove(MemAddr addr) {
    return BaseOp::remove(detail::as_ptr(this->map()), addr);
  }

  template <typename FwdIter, typename Callback>
  inline void remove_range(FwdIter&& s, FwdIter&& e, Callback&& log) {
    BaseOp::template bulk_op<BulkOperation::remove>(detail::as_ptr(this->map()), std::forward<FwdIter>(s),
                                                    std::forward<FwdIter>(e), std::forward<Callback>(log));
  }
};

template <typename BaseOp>
struct SharedMutexMap : public BaseOp {
 private:
  mutable std::shared_mutex alloc_m;

 public:
  [[nodiscard]] inline llvm::Optional<RuntimeT::MapEntry> find(MemAddr addr) const {
    std::shared_lock<std::shared_mutex> guard(alloc_m);
    return BaseOp::find(addr);
  }

  [[nodiscard]] inline bool put(MemAddr addr, const RuntimeT::MappedType& entry) {
    std::lock_guard<std::shared_mutex> guard(alloc_m);
    return BaseOp::put(addr, entry);
  }

  [[nodiscard]] inline llvm::Optional<RuntimeT::MappedType> remove(MemAddr addr) {
    std::lock_guard<std::shared_mutex> guard(alloc_m);
    return BaseOp::remove(addr);
  }

  template <typename FwdIter, typename Callback>
  inline void remove_range(FwdIter&& s, FwdIter&& e, Callback&& log) {
    std::lock_guard<std::shared_mutex> guard(alloc_m);
    BaseOp::remove_range(std::forward<FwdIter>(s), std::forward<FwdIter>(e), std::forward<Callback>(log));
  }
};

#ifdef USE_SAFEPTR
template <typename BaseOp>
struct SafePtrdMap : protected BaseOp {
  [[nodiscard]] inline llvm::Optional<RuntimeT::MapEntry> find(MemAddr addr) const {
    auto slockedAllocs = sf::slock_safe_ptr(this->map());
    return BaseOp::find(slockedAllocs, addr);
  }

  [[nodiscard]] inline bool put(MemAddr addr, const RuntimeT::MappedType& entry) {
    auto guard = sf::xlock_safe_ptr(this->map());
    return BaseOp::put(guard, addr, entry);
  }

  [[nodiscard]] inline llvm::Optional<RuntimeT::MappedType> remove(MemAddr addr) {
    auto guard = sf::xlock_safe_ptr(this->map());
    return BaseOp::remove(guard, addr);
  }

  template <typename FwdIter, typename Callback>
  inline void remove_range(FwdIter&& s, FwdIter&& e, Callback&& log) {
    using namespace detail;
    auto guard = sf::xlock_safe_ptr(this->map());
    BaseOp::template bulk_op<BulkOperation::remove>(guard, std::forward<FwdIter>(s), std::forward<FwdIter>(e),
                                                    std::forward<Callback>(log));
  }
};
#endif
}  // namespace mixin

#ifdef USE_SAFEPTR
using PointerMap = mixin::SafePtrdMap<mixin::MapOp>;
#else
#ifdef TYPEART_DISABLE_THREAD_SAFETY
using PointerMap = mixin::StandardMapBase<mixin::MapOp>;
#else
using PointerMap = mixin::SharedMutexMap<mixin::StandardMapBase<mixin::MapOp>>;
#endif
#endif

}  // namespace typeart

#endif  // TYPEART_ALLOCMAPWRAPPER_H

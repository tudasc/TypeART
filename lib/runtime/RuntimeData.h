// TypeART library
//
// Copyright (c) 2017-2025 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#ifndef TYPEART_RUNTIMEDATA_H
#define TYPEART_RUNTIMEDATA_H

#ifdef TYPEART_BTREE
#error TypeART-RT: TYPART_BTREE is deprecated.
#endif

#ifdef TYPEART_ABSEIL
#ifdef TYPEART_PHMAP
#error TypeART-RT: Set ABSL and PHMAP, mutually exclusive.
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wshadow"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#pragma GCC diagnostic pop
#endif

#ifdef TYPEART_PHMAP
#ifdef TYPEART_ABSEIL
#error TypeART-RT: Set ABSL and PHMAP, mutually exclusive.
#endif
#include "parallel_hashmap/btree.h"
#include "parallel_hashmap/phmap.h"
#endif

#if !defined(TYPEART_PHMAP) && !defined(TYPEART_ABSEIL)
#include <map>
#include <unordered_map>
#endif

#ifdef USE_SAFEPTR
#ifdef TYPEART_DISABLE_THREAD_SAFETY
#error TypeART-RT: Safe_ptr and disabled thread safety illegal
#endif
#include "safe_ptr.h"
#endif

#if defined(__has_feature)
#if __has_feature(address_sanitizer) && !defined(__SANITIZE_ADDRESS__)
#define __SANITIZE_ADDRESS__ 1
#endif
#endif

#include <cstddef>  // size_t
#include <cstdint>
#include <vector>

namespace typeart {

using MemAddr = const void*;

struct PointerInfo final {
  int typeId{-1};
  size_t count{0};
  MemAddr debug{nullptr};
};

namespace global_types {
struct GlobalTypeInfoData;
struct GlobalTypeInfo {
  std::int32_t type_id;
  const std::uint32_t extent;
  const GlobalTypeInfoData* data;
};
}  // namespace global_types

struct RuntimeT {
  using Stack = std::vector<MemAddr>;
  static constexpr auto StackReserve{512U};
  static constexpr char StackName[] = "std::vector";
#ifdef TYPEART_PHMAP
  using PointerMapBaseT = phmap::btree_map<MemAddr, PointerInfo>;
  template <class K, class V>
  using HashmapT                  = phmap::flat_hash_map<K, V>;
  static constexpr char MapName[] = "phmap::btree_map";
#endif
#ifdef TYPEART_ABSEIL
  using PointerMapBaseT = absl::btree_map<MemAddr, PointerInfo>;
  template <class K, class V>
#ifdef __SANITIZE_ADDRESS__
  using HashmapT = std::unordered_map<K, V>;
#else
  using HashmapT = absl::flat_hash_map<K, V>;
#endif
  static constexpr char MapName[] = "absl::btree_map";
#endif
#if !defined(TYPEART_PHMAP) && !defined(TYPEART_ABSEIL)
  using PointerMapBaseT = std::map<MemAddr, PointerInfo>;
  template <class K, class V>
  using HashmapT                  = std::unordered_map<K, V>;
  static constexpr char MapName[] = "std::map";
#endif
#ifdef USE_SAFEPTR
  using PointerMap = sf::contfree_safe_ptr<PointerMapBaseT>;
  static constexpr bool has_safe_map{true};
#else
  using PointerMap = PointerMapBaseT;
  static constexpr bool has_safe_map{false};
#endif
  using MapEntry       = PointerMapBaseT::value_type;
  using MappedType     = PointerMapBaseT::mapped_type;
  using MapKey         = PointerMapBaseT::key_type;
  using StackEntry     = Stack::value_type;
  using TypeLookupMapT = HashmapT<MemAddr, int>;
};

}  // namespace typeart

#endif  // TYPEART_RUNTIMEDATA_H

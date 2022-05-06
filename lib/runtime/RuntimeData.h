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
#include "absl/container/btree_map.h"
#pragma GCC diagnostic pop
#endif

#ifdef TYPEART_PHMAP
#ifdef TYPEART_ABSEIL
#error TypeART-RT: Set ABSL and PHMAP, mutually exclusive.
#endif
#include "parallel_hashmap/btree.h"
#endif

#if !defined(TYPEART_PHMAP) && !defined(TYPEART_ABSEIL)
#include <map>
#endif

#ifdef USE_SAFEPTR
#ifdef TYPEART_DISABLE_THREAD_SAFETY
#error TypeART-RT: Safe_ptr and disabled thread safety illegal
#endif
#include "safe_ptr.h"
#endif

#include <cstddef>  // size_t
#include <vector>

namespace typeart {

using MemAddr = const void*;

struct PointerInfo final {
  int typeId{-1};
  size_t count{0};
  MemAddr debug{nullptr};
};

struct RuntimeT {
  using Stack = std::vector<MemAddr>;
  static constexpr auto StackReserve{512U};
  static constexpr char StackName[] = "std::vector";
#ifdef TYPEART_PHMAP
  using PointerMapBaseT           = phmap::btree_map<MemAddr, PointerInfo>;
  static constexpr char MapName[] = "phmap::btree_map";
#endif
#ifdef TYPEART_ABSEIL
  using PointerMapBaseT           = absl::btree_map<MemAddr, PointerInfo>;
  static constexpr char MapName[] = "absl::btree_map";
#endif
#if !defined(TYPEART_PHMAP) && !defined(TYPEART_ABSEIL)
  using PointerMapBaseT           = std::map<MemAddr, PointerInfo>;
  static constexpr char MapName[] = "std::map";
#endif
#ifdef USE_SAFEPTR
  using PointerMap = sf::contfree_safe_ptr<PointerMapBaseT>;
  static constexpr bool has_safe_map{true};
#else
  using PointerMap = PointerMapBaseT;
  static constexpr bool has_safe_map{false};
#endif
  using MapEntry   = PointerMapBaseT::value_type;
  using MappedType = PointerMapBaseT::mapped_type;
  using MapKey     = PointerMapBaseT::key_type;
  using StackEntry = Stack::value_type;
};

}  // namespace typeart

#endif  // TYPEART_RUNTIMEDATA_H

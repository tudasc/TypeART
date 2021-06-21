//
// Created by ahueck on 12.10.20.
//

#ifndef TYPEART_RUNTIMEDATA_H
#define TYPEART_RUNTIMEDATA_H

#ifdef USE_BTREE
#ifdef USE_ABSL
#error TypeART-RT: Set BTREE and ABSL, mutually exclusive.
#endif
#include "btree_map.h"
#endif

#ifdef USE_ABSL
#ifdef USE_BTREE
#error TypeART-RT: Set ABSL and BTREE, mutually exclusive.
#endif
#include "absl/container/btree_map.h"
#endif

#if !defined(USE_BTREE) && !defined(USE_ABSL)
#include <map>
#endif

#ifdef USE_SAFEPTR
#ifdef DISABLE_THREAD_SAFETY
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
#ifdef USE_BTREE
  using PointerMapBaseT           = btree::btree_map<MemAddr, PointerInfo>;
  static constexpr char MapName[] = "btree::btree_map";
#endif
#ifdef USE_ABSL
  using PointerMapBaseT           = absl::btree_map<MemAddr, PointerInfo>;
  static constexpr char MapName[] = "absl::btree_map";
#endif
#if !defined(USE_BTREE) && !defined(USE_ABSL)
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

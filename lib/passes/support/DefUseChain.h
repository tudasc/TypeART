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

#ifndef TYPEART_DEFUSECHAIN_H
#define TYPEART_DEFUSECHAIN_H

#include "support/Logger.h"
#include "support/Util.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Instructions.h"

#include <algorithm>
#include <functional>
#include <optional>

namespace typeart::util {

struct DefUseChain {
 public:
  enum MatchResult { kNoMatch = 0, kMatch, kCancel, kSkip };

 private:
  llvm::SmallVector<llvm::Value*, 16> working_set;
  llvm::SmallPtrSet<llvm::Value*, 16> seen_set;

  void addToWorkS(llvm::Value* v) {
    if (v != nullptr && seen_set.find(v) == seen_set.end()) {
      working_set.push_back(v);
      seen_set.insert(v);
    }
  }

  template <typename Range>
  void addToWork(Range&& vals) {
    for (auto v : vals) {
      addToWorkS(v);
    }
  }

  auto peek() -> llvm::Value* {
    if (working_set.empty()) {
      return nullptr;
    }
    auto user_iter = working_set.end() - 1;
    working_set.erase(user_iter);
    return *user_iter;
  }

  template <typename AllowedTy, typename SDirection, typename CallBackF>
  void do_traverse(SDirection&& search, llvm::Value* start, CallBackF match) {
    const auto should_search = [](auto user) {
      return llvm::isa<AllowedTy>(user) && !llvm::isa<llvm::ConstantData>(user);
    };
    const auto apply_search = [&](auto val) {
      auto value = search(val);
      if (value) {
        addToWork(value.value());
      }
    };

    apply_search(start);

    while (!working_set.empty()) {
      auto user = peek();
      if (user == nullptr) {
        continue;
      }
      if (MatchResult m = match(user); m != kNoMatch) {
        switch (m) {
          case kSkip:
            continue;
          case kCancel:
            break;
          default:
            break;
        }
      }
      if (should_search(user)) {
        apply_search(user);
      }
    }
    working_set.clear();
    seen_set.clear();
  }

 public:
  template <typename CallBackF>
  void traverse(llvm::Value* start, CallBackF&& match) {
    LOG_DEBUG("Start traversal for value: " << util::dump(*start));
    do_traverse<llvm::Value>([](auto val) -> std::optional<decltype(val->users())> { return val->users(); }, start,
                             std::forward<CallBackF>(match));
    LOG_DEBUG("Finished traversal");
  }

  template <typename Search, typename CallBackF>
  void traverse_custom(llvm::Value* start, Search&& s, CallBackF&& match) {
    LOG_DEBUG("Start traversal for value: " << util::dump(*start));
    do_traverse<llvm::Value>(std::forward<Search>(s), start, std::forward<CallBackF>(match));
    LOG_DEBUG("Finished traversal");
  }

  template <typename CallBackF>
  void traverse_with_store(llvm::Value* start, CallBackF&& match) {
    LOG_DEBUG("Start traversal for value: " << util::dump(*start));
    do_traverse<llvm::Value>(
        [](auto val) -> std::optional<decltype(val->users())> {
          if (auto cinst = llvm::dyn_cast<llvm::StoreInst>(val)) {
            return cinst->getPointerOperand()->users();
          }
          return val->users();
        },
        start, std::forward<CallBackF>(match));
    LOG_DEBUG("Finished traversal");
  }
};

}  // namespace typeart::util

#endif  // TYPEART_DEFUSECHAIN_H

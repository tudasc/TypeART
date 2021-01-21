//
// Created by ahueck on 21.01.21.
//

#ifndef TYPEART_DEFUSECHAIN_H
#define TYPEART_DEFUSECHAIN_H

using namespace llvm;

#include "support/Logger.h"
#include "support/Util.h"

#include <algorithm>
#include <functional>
#include <llvm/IR/Instructions.h>

namespace typeart::util {

struct DefUseChain {
 public:
  enum MatchResult { no_match = 0, match, cancel, skip };

 private:
  llvm::SmallVector<Value*, 16> working_set;
  llvm::SmallPtrSet<Value*, 16> seen_set;

  void addToWorkS(Value* v) {
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

  auto peek() -> Value* {
    if (working_set.empty()) {
      return nullptr;
    }
    auto user_iter = working_set.end() - 1;
    working_set.erase(user_iter);
    return *user_iter;
  }

  template <typename AllowedTy, typename SDirection, typename CallBackF>
  void do_traverse(SDirection&& search, Value* start, CallBackF match) {
    const auto should_search = [](auto user) { return isa<AllowedTy>(user) && !isa<ConstantData>(user); };
    const auto apply_search  = [&](auto val) {
      auto value = search(val);
      if (value) {
        addToWork(value.getValue());
      }
    };

    apply_search(start);

    while (!working_set.empty()) {
      auto user = peek();
      if (user == nullptr) {
        continue;
      }
      LOG_DEBUG("Analyzing value: " << *user);
      if (MatchResult m = match(user); m != no_match) {
        switch (m) {
          case skip:
            LOG_DEBUG("...skip from match");
            continue;
          case cancel:
            LOG_DEBUG("...cancel from match");
            break;
          default:
            LOG_DEBUG("...continue from match")
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
  void traverse(Value* start, CallBackF&& match) {
    LOG_DEBUG("Start search for value: " << util::dump(*start));
    do_traverse<Value>([](auto val) -> Optional<decltype(val->users())> { return val->users(); }, start,
                       std::forward<CallBackF>(match));
    LOG_DEBUG("Finished search");
  }
};

}  // namespace typeart::util

#endif  // TYPEART_DEFUSECHAIN_H

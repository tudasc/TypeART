//
// Created by ahueck on 21.10.20.
//

#ifndef TYPEART_FILTERBASE_H
#define TYPEART_FILTERBASE_H

#include "../support/Logger.h"
#include "../support/Util.h"
#include "Filter.h"

#include "llvm/IR/CallSite.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"

#include <type_traits>

namespace typeart::filter {

enum class FilterAnalysis { cont = 0, nofilter, filter };

using namespace llvm;

struct DefUseQueue {
  llvm::SmallPtrSet<Value*, 16> visited_set;
  llvm::SmallVector<Value*, 16> working_set;
  llvm::SmallVector<CallSite, 8> working_set_calls;

  explicit DefUseQueue(Value* init) {
    working_set.emplace_back(init);
  }

  void reset() {
    visited_set.clear();
    working_set.clear();
    working_set_calls.clear();
  }

  bool empty() const {
    return working_set.empty();
  }

  void addToWorkS(Value* v) {
    if (v != nullptr && visited_set.find(v) == visited_set.end()) {
      working_set.push_back(v);
      visited_set.insert(v);
    }
  }

  template <typename Range>
  void addToWork(Range&& values) {
    for (auto v : values) {
      addToWorkS(v);
    }
  }

  Value* peek() {
    if (working_set.empty()) {
      return nullptr;
    }
    auto user_iter = working_set.end() - 1;
    working_set.erase(user_iter);
    return *user_iter;
  }
};

struct DefaultSearch {};

struct SearchStoreDir {
  auto search(Value* val) -> Optional<decltype(val->users())> {
    if (auto store = llvm::dyn_cast<StoreInst>(val)) {
      val = store->getPointerOperand();
      if (llvm::isa<AllocaInst>(val)) {
        return None;
      }
    }
    return val->users();
  }
};

template <typename CallSiteHandler, typename Search = DefaultSearch>
class BaseFilter : public Filter {
  CallSiteHandler handler;
  Search search_dir;
  bool malloc_mode{false};
  llvm::Function* start_f{nullptr};

 public:
  BaseFilter(const std::string& glob) : handler(glob), search_dir() {
  }

  bool filter(llvm::Value* in) override {
    if (in == nullptr) {
      LOG_DEBUG("Called with nullptr");
      return false;
    }

    /* TODO if(recursion) stop; */

    DefUseQueue queue(in);

    while (!queue.empty()) {
      auto val = queue.peek();
      if (!val) {
        continue;
      }

      CallSite site(val);
      if (site.isCall()) {
        const auto callee        = site.getCalledFunction();
        const bool indirect_call = callee == nullptr;

        // Indirect calls (sth. like function pointers)
        if (indirect_call) {
          if constexpr (CallSiteHandler::Trait::Indirect::value) {
            auto status = handler.indirect(in, val, site);
            if (FilterAnalysis::nofilter == status) {
              return false;
            }
          } else {
            return false;
          }
        }

        const bool is_decl      = callee->isDeclaration();
        const bool is_intrinsic = site.getIntrinsicID() != Intrinsic::not_intrinsic;

        // Handle decl
        if (is_decl) {
          if (is_intrinsic) {
            continue;  // TODO make that pluggable
          }
          // Handle decl (like MPI calls)
          auto status = handler.decl(in, val, site);
          if (FilterAnalysis::nofilter == status) {
            return false;
          } else if (FilterAnalysis::cont == status) {
            continue;
          }
        }

        // Handle definitions
        auto status = handler.def(in, val, site);
        if (FilterAnalysis::nofilter == status) {
          return false;
        } else if (FilterAnalysis::cont == status) {
          continue;
        }

        // TODO handle across function dataflow
      }

      // Look forward at dataflow
      if constexpr (std::is_same_v<Search, DefaultSearch>) {
        queue.addToWork(val->users());
      } else {
        auto values = search_dir.search(val);
        if (values) {
          queue.addToWork(values.getValue());
        }
      }
    }

    return false;
  }

  virtual void setStartingFunction(llvm::Function* f) override {
    start_f = f;
  };

  virtual void setMode(bool m) override {
    malloc_mode = m;
  };
};

struct FilterTrait {
  using Indirect  = std::true_type::type;
  using Intrinsic = std::false_type::type;
};

struct Handler {
  using Trait = FilterTrait;

  std::string filter;

  Handler(std::string filter) : filter(filter) {
  }

  FilterAnalysis indirect(Value* in, Value* current, CallSite c) {
    return FilterAnalysis::nofilter;
  }

  FilterAnalysis decl(Value* in, Value* current, CallSite c) {
    return FilterAnalysis::nofilter;
  }

  FilterAnalysis def(Value* in, Value* current, CallSite c) {
    return FilterAnalysis::nofilter;
  }
};

using StandardForwardFilter = BaseFilter<Handler, SearchStoreDir>;

}  // namespace typeart::filter

#endif  // TYPEART_FILTERBASE_H

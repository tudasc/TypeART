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
  explicit BaseFilter(const std::string& glob) : handler(glob), search_dir() {
  }

  explicit BaseFilter(const CallSiteHandler& handler) : handler(handler), search_dir() {
  }

  bool filter(llvm::Value* in) override {
    if (in == nullptr) {
      LOG_DEBUG("Called with nullptr");
      return false;
    }

    /* TODO if(recursion) stop; */

    /* do a pre-flow tracking check of value in  */
    if constexpr (CallSiteHandler::Support::PreCheck) {
      auto status = handler.precheck(in);
      if (FilterAnalysis::filter == status) {
        return true;
      }
    } else {
    }

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
          if constexpr (CallSiteHandler::Support::Indirect) {
            auto status = handler.indirect(in, val);
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
            if constexpr (CallSiteHandler::Support::Intrinsic) {
              auto status = handler.intrinsic(in, val);
              if (FilterAnalysis::nofilter == status) {
                return false;
              }
            } else {
              return false;
            }
          }

          // Handle decl (like MPI calls)
          if constexpr (CallSiteHandler::Support::Declaration) {
            auto status = handler.decl(in, val);
            if (FilterAnalysis::nofilter == status) {
              return false;
            } else if (FilterAnalysis::cont == status) {
              continue;
            }
          } else {
            return false;
          }
        } else {
          // Handle definitions
          if constexpr (CallSiteHandler::Support::Definition) {
            auto status = handler.def(in, val);
            if (FilterAnalysis::nofilter == status) {
              return false;
            } else if (FilterAnalysis::cont == status) {
              continue;
            }
          } else {
            return false;
          }
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
  constexpr static bool Indirect    = true;
  constexpr static bool Intrinsic   = false;
  constexpr static bool Declaration = true;
  constexpr static bool Definition  = true;
  constexpr static bool PreCheck    = false;
};

struct Handler {
  using Support = FilterTrait;

  std::string filter;

  Handler(std::string filter) : filter(std::move(filter)) {
  }

  FilterAnalysis indirect(Value* in, Value* current) {
    return FilterAnalysis::nofilter;
  }

  FilterAnalysis decl(Value* in, Value* current) {
    return FilterAnalysis::nofilter;
  }

  FilterAnalysis def(Value* in, Value* current) {
    return FilterAnalysis::nofilter;
  }
};

using StandardForwardFilter = BaseFilter<Handler, SearchStoreDir>;

}  // namespace typeart::filter

#endif  // TYPEART_FILTERBASE_H

//
// Created by ahueck on 21.10.20.
//

#ifndef TYPEART_FILTERBASE_H
#define TYPEART_FILTERBASE_H

#include "../support/Logger.h"
#include "../support/Util.h"
#include "Filter.h"
#include "FilterUtil.h"
#include "IRPath.h"

#include "llvm/IR/CallSite.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"

#include <iterator>
#include <type_traits>

namespace typeart::filter {

using namespace llvm;

enum class FilterAnalysis { skip = 0, cont, keep, filter };

struct DefaultSearch {
  auto search(Value* val) -> Optional<decltype(val->users())> {
    return val->users();
  }
};

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

  template <typename... Args>
  BaseFilter(Args&&... args) : handler(std::forward<Args>(args)...) {
  }

  bool filter(llvm::Value* in) override {
    if (in == nullptr) {
      LOG_DEBUG("Called with nullptr");
      return false;
    }

    /* TODO if(recursion) stop; */

    /* do a pre-flow tracking check of value in  */
    if constexpr (CallSiteHandler::Support::PreCheck) {
      auto status = handler.precheck(in, start_f);
      switch (status) {
        case FilterAnalysis::filter:
          return true;
        case FilterAnalysis::keep:
          return false;
        case FilterAnalysis::skip:
          [[fallthrough]];
        case FilterAnalysis::cont:
          [[fallthrough]];
        default:
          break;
      }
    } else {
    }

    Path p;
    return DFSfilter(in, p);
  }

  virtual void setStartingFunction(llvm::Function* f) override {
    start_f = f;
  };

  virtual void setMode(bool m) override {
    malloc_mode = m;
  };

 private:
  bool DFSfilter(llvm::Value* current, Path& path) {
    path.push(current);

    LOG_DEBUG(path);

    bool skip{false};
    // In-order analysis
    auto status = callsite(current, path);
    switch (status) {
      case FilterAnalysis::keep:
        path.pop();
        return false;
      case FilterAnalysis::skip:
        skip = true;
      default:
        break;
    }

    auto succs = search_dir.search(current);
    if (succs && !skip) {
      for (auto* successor : succs.getValue()) {
        if (path.contains(successor)) {
          // Avoid recursion (e.g., with store inst pointer operands pointing to an allocation)
          continue;
        }
        const auto filter = DFSfilter(successor, path);
        if (!filter) {
          path.pop();
          return false;
        }
      }
    }

    path.pop();
    return true;
  }

  FilterAnalysis callsite(llvm::Value* val, const Path& path) {
    CallSite site(val);
    if (site.isCall()) {
      const auto callee        = site.getCalledFunction();
      const bool indirect_call = callee == nullptr;

      // Indirect calls (sth. like function pointers)
      if (indirect_call) {
        if constexpr (CallSiteHandler::Support::Indirect) {
          auto status = handler.indirect(site, path);
          LOG_DEBUG("Indirect call.")
          return status;
        } else {
          LOG_DEBUG("Indirect call, keep.")
          return FilterAnalysis::keep;
        }
      }

      const bool is_decl      = callee->isDeclaration();
      const bool is_intrinsic = site.getIntrinsicID() != Intrinsic::not_intrinsic;

      // Handle decl
      if (is_decl) {
        if (is_intrinsic) {
          if constexpr (CallSiteHandler::Support::Intrinsic) {
            auto status = handler.intrinsic(site, path);
            LOG_DEBUG("Intrinsic call.")
            return status;
          } else {
            LOG_DEBUG("Skip intrinsic.")
            return FilterAnalysis::skip;
          }
        }

        // Handle decl (like MPI calls)
        if constexpr (CallSiteHandler::Support::Declaration) {
          auto status = handler.decl(site, path);
          LOG_DEBUG("Decl call.")
          return status;
        } else {
          LOG_DEBUG("Declaration, keep.")
          return FilterAnalysis::keep;
        }
      } else {
        // Handle definitions
        if constexpr (CallSiteHandler::Support::Definition) {
          auto status = handler.def(site, path);
          LOG_DEBUG("Defined call.")
          return status;
        } else {
          LOG_DEBUG("Definition, keep.")
          return FilterAnalysis::keep;
        }
      }
    }
    return FilterAnalysis::cont;
  }
};

}  // namespace typeart::filter

#endif  // TYPEART_FILTERBASE_H

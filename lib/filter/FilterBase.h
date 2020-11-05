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

enum class FilterAnalysis { skip = 0, cont, keep, filter, follow };

struct DefaultSearch {
  auto search(Value* val) -> Optional<decltype(val->users())> {
    if (auto store = llvm::dyn_cast<StoreInst>(val)) {
      val = store->getPointerOperand();
      if (llvm::isa<AllocaInst>(val) && !store->getValueOperand()->getType()->isPointerTy()) {
        // 1. if we store to an alloca, and the value is not a pointer (i.e., a value) there is no connection to follow
        // w.r.t. dataflow. (TODO exceptions could be some pointer arithm.)
        return None;
      }
      // 2. TODO if we store to a pointer, analysis is required to filter simple aliasing pointer (filter opportunity,
      // see test 01_alloca.llin variable a and c -- c points to a, then c gets passed to MPI)
      // 2.1 care has to be taken for argument store to aliasing local (implicit) alloc, i.e., see same test variable %x
      // passed to func foo_bar2
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

    FPath fpath(start_f);
    return DFSFuncFilter(in, fpath);
  }

  virtual void setStartingFunction(llvm::Function* f) override {
    start_f = f;
  };

  virtual void setMode(bool m) override {
    malloc_mode = m;
  };

 private:
  bool DFSFuncFilter(llvm::Value* current, FPath& fpath) {
    auto f = fpath.getCurrentFunc();
    if (!f) {
      return false;
    }

    [[maybe_unused]] llvm::Function* currentF = f.getValue();

    /* do a pre-flow tracking check of value in  */
    if constexpr (CallSiteHandler::Support::PreCheck) {
      auto status = handler.precheck(current, currentF);
      switch (status) {
        case FilterAnalysis::filter:
          fpath.pop();
          return true;
        case FilterAnalysis::keep:
          fpath.pop();
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

    PathList defPath;  // pathes that reach a definition in currentF
    Path p;
    const auto filter = DFSfilter(current, p, defPath);

    if (!filter) {
      return false;
    }

    if (defPath.empty()) {
      return filter;
    }

    for (auto& path2def : defPath) {
      auto csite = path2def.getEnd();
      if (!csite) {
        continue;
      }
      llvm::CallSite c(csite.getValue());
      if (fpath.contains(c)) {
        continue;
      }

      fpath.push(path2def);

      LOG_DEBUG(fpath);

      auto argv = args(c, path2def);

      if (argv.size() > 1) {
        LOG_DEBUG("All args are looked at.")
      }

      for (auto& arg : argv) {
        const auto dfs_filter = DFSFuncFilter(arg, fpath);
        if (!dfs_filter) {
          fpath.pop();
          return false;
        }
      }
    }

    fpath.pop();
    return true;
  }

  bool DFSfilter(llvm::Value* current, Path& path, PathList& plist) {
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
        break;
      case FilterAnalysis::follow:
        LOG_DEBUG("Analyze definition in path");
        // store path (with the callsite) for a function recursive check later
        plist.emplace_back(path);
        break;
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
        const auto filter = DFSfilter(successor, path, plist);
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

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
#include "IRSearch.h"

#include "llvm/IR/CallSite.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"

#include <iterator>
#include <type_traits>

namespace typeart::filter {

enum class FilterAnalysis {
  Skip = 0,   // Do not follow users of current decl/def etc.
  Continue,   // Continue searching users of decl/def etc.
  Keep,       // Keep the value (return false)
  Filter,     // Filter the value (return true)
  FollowDef,  // Want analysis of the called function def
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
      LOG_WARNING("Called with nullptr");
      return false;
    }

    FPath fpath(start_f);
    const auto filter = DFSFuncFilter(in, fpath);
    if (!filter) {
      LOG_DEBUG(fpath);
    }
    return filter;
  }

  virtual void setStartingFunction(llvm::Function* f) override {
    start_f = f;
  };

  virtual void setMode(bool m) override {
    malloc_mode = m;
  };

 private:
  bool DFSFuncFilter(llvm::Value* current, FPath& fpath) {
    // is null in case of global:
    llvm::Function* currentF = fpath.getCurrentFunc();

    /* do a pre-flow tracking check of value in  */
    if constexpr (CallSiteHandler::Support::PreCheck) {
      if (currentF != nullptr) {
        auto status = handler.precheck(current, currentF);
        switch (status) {
          case FilterAnalysis::Filter:
            fpath.pop();
            LOG_DEBUG("Pre-check, filter")
            return true;
          case FilterAnalysis::Keep:
            LOG_DEBUG("Pre-check, keep")
            return false;
          case FilterAnalysis::Skip:
            [[fallthrough]];
          case FilterAnalysis::Continue:
            [[fallthrough]];
          default:
            break;
        }
      }
    } else {
    }

    PathList defPath;  // pathes that reach a definition in currentF
    Path p;
    const auto filter = DFSfilter(current, p, defPath);

    if (!filter) {
      // for diagnostic output, store the last path
      fpath.pushFinal(p);
      return false;
    }

    for (auto& path2def : defPath) {
      auto csite = path2def.getEnd();
      if (!csite) {
        continue;
      }

      llvm::CallSite c(csite.getValue());
      if (fpath.contains(c)) {
        // Avoid recursion:
        // TODO a continue may be wrong, if the function itself eventually calls "MPI"?
        continue;
      }

      fpath.push(path2def);

      auto argv = args(c, path2def);
      if (argv.size() > 1) {
        LOG_DEBUG("All args are looked at.")
      }

      for (auto* arg : argv) {
        if (arg == nullptr) {
          LOG_FATAL("Called with nullptr. " << c.getCalledFunction()->getName());
          return false;
        }
        const auto dfs_filter = DFSFuncFilter(arg, fpath);
        if (!dfs_filter) {
          return false;
        }
      }
    }

    fpath.pop();
    return true;
  }

  bool DFSfilter(llvm::Value* current, Path& path, PathList& plist) {
    if (current == nullptr) {
      LOG_FATAL("Called with nullptr: " << path);
      return false;
    }

    path.push(current);

    bool skip{false};
    // In-order analysis
    auto status = callsite(current, path);
    switch (status) {
      case FilterAnalysis::Keep:
        return false;
      case FilterAnalysis::Skip:
        skip = true;
        break;
      case FilterAnalysis::FollowDef:
        LOG_DEBUG("Analyze definition in path");
        // store path (with the callsite) for a function recursive check later
        plist.emplace_back(path);
        break;
      default:
        break;
    }

    auto succs = search_dir.search(current, path);
    if (succs && !skip) {
      for (auto* successor : succs.getValue()) {
        if (path.contains(successor)) {
          // Avoid recursion (e.g., with store inst pointer operands pointing to an allocation)
          continue;
        }
        const auto filter = DFSfilter(successor, path, plist);
        if (!filter) {
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
          LOG_DEBUG("Indirect call: " << util::try_demangle(site))
          return status;
        } else {
          LOG_DEBUG("Indirect call, keep: " << util::try_demangle(site))
          return FilterAnalysis::Keep;
        }
      }

      const bool is_decl      = callee->isDeclaration();
      const bool is_intrinsic = site.getIntrinsicID() != Intrinsic::not_intrinsic;

      // Handle decl
      if (is_decl) {
        if (is_intrinsic) {
          if constexpr (CallSiteHandler::Support::Intrinsic) {
            auto status = handler.intrinsic(site, path);
            LOG_DEBUG("Intrinsic call: " << util::try_demangle(site))
            return status;
          } else {
            LOG_DEBUG("Skip intrinsic: " << util::try_demangle(site))
            return FilterAnalysis::Skip;
          }
        }

        // Handle decl (like MPI calls)
        if constexpr (CallSiteHandler::Support::Declaration) {
          auto status = handler.decl(site, path);
          LOG_DEBUG("Decl call: " << util::try_demangle(site))
          return status;
        } else {
          LOG_DEBUG("Declaration, keep: " << util::try_demangle(site))
          return FilterAnalysis::Keep;
        }
      } else {
        // Handle definitions
        if constexpr (CallSiteHandler::Support::Definition) {
          auto status = handler.def(site, path);
          LOG_DEBUG("Defined call: " << util::try_demangle(site))
          return status;
        } else {
          LOG_DEBUG("Definition, keep: " << util::try_demangle(site))
          return FilterAnalysis::Keep;
        }
      }
    }
    return FilterAnalysis::Continue;
  }
};

}  // namespace typeart::filter

#endif  // TYPEART_FILTERBASE_H

//
// Created by ahueck on 21.10.20.
//

#ifndef TYPEART_FILTERBASE_H
#define TYPEART_FILTERBASE_H

#include "../support/Logger.h"
#include "../support/Util.h"
#include "Filter.h"
#include "FilterUtil.h"

#include "llvm/IR/CallSite.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"

#include <llvm/ADT/DepthFirstIterator.h>
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
            auto status = handler.indirect(in, site);
            switch (status) {
              case FilterAnalysis::keep:
                return false;
              case FilterAnalysis::cont:
                break;
              default:
                LOG_DEBUG("Indirect call, continuing analysis.");
                continue;
            }
          } else {
            LOG_DEBUG("Indirect call, filtering.")
            return false;
          }
        }

        const bool is_decl      = callee->isDeclaration();
        const bool is_intrinsic = site.getIntrinsicID() != Intrinsic::not_intrinsic;

        // Handle decl
        if (is_decl) {
          if (is_intrinsic) {
            if constexpr (CallSiteHandler::Support::Intrinsic) {
              auto status = handler.intrinsic(in, site);
              switch (status) {
                case FilterAnalysis::keep:
                  return false;
                case FilterAnalysis::cont:
                  break;
                default:
                  LOG_DEBUG("Skip Intrinsic call, continuing analysis.");
                  continue;
              }
            } else {
              LOG_DEBUG("Skip intrinsic.")
              continue;
            }
          }

          // Handle decl (like MPI calls)
          if constexpr (CallSiteHandler::Support::Declaration) {
            auto status = handler.decl(in, site);
            switch (status) {
              case FilterAnalysis::keep:
                return false;
              case FilterAnalysis::cont:
                break;
              default:
                LOG_DEBUG("Decl call, continuing analysis.");
                continue;
            }
          } else {
            LOG_DEBUG("Declaration, filter.")
            return false;
          }
        } else {
          // Handle definitions
          if constexpr (CallSiteHandler::Support::Definition) {
            auto status = handler.def(in, site);
            switch (status) {
              case FilterAnalysis::keep:
                return false;
              case FilterAnalysis::cont:
                break;
              default:
                LOG_DEBUG("Defined call, continuing analysis.");
                continue;
            }
          } else {
            LOG_DEBUG("Definition, filter.")
            return false;
          }
        }
      }

      // Look forward at dataflow
      auto values = search_dir.search(val);
      if (values) {
        queue.addToWork(values.getValue());
      }
    }

    return true;  // no early exit, we should filter
  }

  virtual void setStartingFunction(llvm::Function* f) override {
    start_f = f;
  };

  virtual void setMode(bool m) override {
    malloc_mode = m;
  };
};

}  // namespace typeart::filter

#endif  // TYPEART_FILTERBASE_H

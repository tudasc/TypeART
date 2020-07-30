//
// Created by ahueck on 28.07.20.
//

#ifndef TYPEART_CGFILTER_H
#define TYPEART_CGFILTER_H

#include "Filter.h"
#include "support/CGInterface.h"
#include "support/Logger.h"
#include "support/Util.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"

namespace typeart {
namespace filter {
using namespace llvm;
class CGFilterImpl final : public FilterBase {
  // Holds pointer to a CG implementation
  std::unique_ptr<CGInterface> callGraph;
  // CGInterface* callGraph{nullptr};

 public:
  explicit CGFilterImpl(const std::string& glob, bool CallFilterDeep, std::string file)
      : FilterBase(glob, CallFilterDeep) {
    if (!callGraph && !file.empty()) {
      LOG_FATAL("Resetting the CGInterface with JSON CG");
      // callGraph.reset(new JSONCG(JSONCG::getJSON(ClCGFile.getValue())));
      callGraph.reset(JSONCG::getJSON(file));
      // callGraph = JSONCG::getJSON(file);
    } else {
      LOG_FATAL("CG File not found " << file);
    }
  }

  bool filter(Value* in) override {
    if (in == nullptr) {
      LOG_DEBUG("Called with nullptr");
      return false;
    }

    if (depth == 15) {
      return false;
    }

    const auto do_cg = [&](auto from) {
      if (callGraph) {
        return callGraph->reachable(from->getName(), call_regex);
      } else {
        return CGInterface::ReachabilityResult::maybe_reaches;
      }
    };

    const auto match = [&](auto callee) -> bool {
      const auto name = getName(callee);
      return util::regex_matches(call_regex, name);
    };

    llvm::SmallPtrSet<Value*, 16> visited_set;
    llvm::SmallVector<Value*, 16> working_set;
    llvm::SmallVector<CallSite, 8> working_set_calls;

    const auto addToWork = [&visited_set, &working_set](auto vals) {
      for (auto v : vals) {
        if (visited_set.find(v) == visited_set.end()) {
          working_set.push_back(v);
          visited_set.insert(v);
        }
      }
    };

    const auto peek = [&working_set]() -> Value* {
      auto user_iter = working_set.end() - 1;
      working_set.erase(user_iter);
      return *user_iter;
    };

    // Seed working set with users of value (e.g., our AllocaInst)
    addToWork(in->users());

    // Search through all users of users of .... (e.g., our AllocaInst)
    while (!working_set.empty()) {
      auto val = peek();

      // If we encounter a callsite, we want to analyze later, or quit in case we have a regex match
      CallSite c(val);
      if (c.isCall()) {
        const auto callee        = c.getCalledFunction();
        const bool indirect_call = callee == nullptr;

        if (indirect_call) {
          LOG_DEBUG("Found an indirect call, not filtering alloca: " << util::dump(*val));
          append_trace("Indirect call");
          return false;  // Indirect calls might contain critical function calls.
        }

        const bool is_decl = callee->isDeclaration();
        // FIXME the MPI calls are all hitting this branch (obviously)
        if (is_decl) {
          LOG_DEBUG("Found call with declaration only. Call: " << util::dump(*c.getInstruction()));
          // append_trace("Decl found ") << util::dump(*c.getInstruction());
          if (c.getIntrinsicID() == Intrinsic::not_intrinsic /*Intrinsic::ID::not_intrinsic*/) {
            if (CallFilterDeep && match(callee) && shouldContinue(c, in)) {
              append_trace("Match, continue: ") << util::dump(*c.getInstruction());
              continue;
            }
            if (match(callee)) {
              append_trace("Pattern ") << call_regex << " match of " << util::dump(*c.getInstruction());
            } else {
              const auto reached = do_cg(c.getCalledFunction());
              if (reached == CGInterface::ReachabilityResult::reaches) {
                append_trace("CG calls pattern ") << getName(c.getCalledFunction());
                return false;
              } else if (reached == CGInterface::ReachabilityResult::never_reaches) {
                append_trace("CG success ") << getName(c.getCalledFunction());
                continue;
              } else if (reached == CGInterface::ReachabilityResult::maybe_reaches) {
                append_trace("CG maybe reaches") << getName(c.getCalledFunction());
                return false; // XXX This should be where we can change semantics
              } else {
                append_trace("CG warn: code path should not be executed ") << getName(c.getCalledFunction());
              }
            }
            return false;
          } else {
            LOG_DEBUG("Call is an intrinsic. Continue analyzing...")
            continue;
          }
        }

        if (match(callee)) {
          LOG_DEBUG("Found a call. Call: " << util::dump(*c.getInstruction()));
          if (CallFilterDeep && shouldContinue(c, in)) {
            continue;
          }
          append_trace("match call");
          return false;
        }

        working_set_calls.push_back(c);
        // Caveat: below at the end of the loop, we add users of the function call to the search even though it might be
        // a simple "sink" for the alloca we analyse
      } else if (auto store = llvm::dyn_cast<StoreInst>(val)) {
        // If we encounter a store, we follow the store target pointer.
        // More inclusive than strictly necessary in some cases.
        LOG_DEBUG("Store found: " << util::dump(*store)
                                  << " Store target has users: " << util::dump(store->getPointerOperand()->users()));
        auto store_target = store->getPointerOperand();
        // FIXME here we check store operand, if target is another alloca, we already track that?:
        // Note: if we apply this to malloc filtering, this might become problematic?
        if (!malloc_mode && llvm::isa<AllocaInst>(store_target)) {
          LOG_DEBUG("Target is alloca, skipping!");
        } else {
          addToWork(store_target->users());
        }
        continue;
      }
      // cont. our search
      addToWork(val->users());
    }
    ++depth;
    const auto filter_callsite =
        std::all_of(working_set_calls.begin(), working_set_calls.end(), [&](CallSite c) { return filter(c, in); });
    if (filter_callsite && !working_set_calls.empty()) {
      append_trace("All calls true ") << working_set.size() << " " << *in;
    } else if (filter_callsite) {
      append_trace("Default filter (no call)");
    }
    return filter_callsite;
  }

 private:
  bool filter(CallSite& csite, Value* in) {
    append_trace("Match call: ") << util::demangle(csite.getCalledFunction()->getName()) << " :: " << *in;
    const auto analyse_arg = [&](auto& csite, auto argNum) -> bool {
      Argument& the_arg = *(csite.getCalledFunction()->arg_begin() + argNum);
      LOG_DEBUG("Calling filter with inst of argument: " << util::dump(the_arg));
      const bool filter_arg = filter(&the_arg);
      LOG_DEBUG("Should filter? : " << filter_arg);
      return filter_arg;
    };

    LOG_DEBUG("Analyzing function call " << csite.getCalledFunction()->getName());

    if (csite.getCalledFunction() == start_f) {
      append_trace("a recursion");
      return true;
    }

    // this only works if we can correlate alloca with argument:
    const auto pos = std::find_if(csite.arg_begin(), csite.arg_end(),
                                  [&in](const Use& arg_use) -> bool { return arg_use.get() == in; });
    // auto pos = csite.arg_end();
    if (pos != csite.arg_end()) {
      const auto argNum = std::distance(csite.arg_begin(), pos);
      LOG_DEBUG("Found exact position: " << argNum);

      const auto arg_ = analyse_arg(csite, argNum);
      if (arg_) {
        append_trace("exact arg pos");
      }
      return arg_;
    } else {
      LOG_DEBUG("Analyze all args, cannot correlate alloca with arg.");

      const auto all_pos = std::all_of(csite.arg_begin(), csite.arg_end(), [&csite, &analyse_arg](const Use& arg_use) {
        auto argNum = csite.getArgumentNo(&arg_use);
        return analyse_arg(csite, argNum);
      });
      if (all_pos) {
        append_trace("all args pos");
      }
      return all_pos;
    }

    append_trace("blanked callsite allows");
    return true;
  }

  bool filter(Argument* arg) {
    for (auto* user : arg->users()) {
      LOG_DEBUG("Looking at arg user " << util::dump(*user));
      // This code is for non mem2reg code (i.e., where the argument is stored to a local alloca):
      if (auto store = llvm::dyn_cast<StoreInst>(user)) {
        // if (auto* alloca = llvm::dyn_cast<AllocaInst>(store->getPointerOperand())) {
        //  LOG_DEBUG("Argument is a store inst and the operand is alloca");
        return filter(store->getPointerOperand());
        // }
      }
    }
    return filter(llvm::dyn_cast<Value>(arg));
  }

  bool shouldContinue(CallSite c, Value* in) const {
    LOG_DEBUG("Found a name match, analyzing closer...");
    const auto is_void_ptr = [](Type* type) {
      return type->isPointerTy() && type->getPointerElementType()->isIntegerTy(8);
    };
    const auto arg_pos = llvm::find_if(c.args(), [&in](const Use& arg_use) -> bool { return arg_use.get() == in; });
    if (arg_pos == c.arg_end()) {
      // we had no direct correlation for the arg position
      // Now checking if void* is passed, if not we can potentially filter!
      auto count_void_ptr = llvm::count_if(c.args(), [&is_void_ptr](const auto& arg) {
        const auto type = arg->getType();
        return is_void_ptr(type);
      });
      if (count_void_ptr > 0) {
        LOG_DEBUG("Call takes a void*, filtering.");
        return false;
      }
      LOG_DEBUG("Call has no void* argument");
    } else {
      // We have an arg_pos match
      const auto argNum = std::distance(c.arg_begin(), arg_pos);
      Argument& the_arg = *(c.getCalledFunction()->arg_begin() + argNum);
      auto type         = the_arg.getType();
      if (is_void_ptr(type)) {
        LOG_DEBUG("Call arg is a void*, filtering.");
        return false;
      }
      LOG_DEBUG("Value* in is not passed as void ptr");
    }
    LOG_DEBUG("No filter necessary for this call, continue.");
    return true;
  }

 public:
  virtual ~CGFilterImpl() = default;
};
}  // namespace filter
}  // namespace typeart
#endif  // TYPEART_CGFILTER_H

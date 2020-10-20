//
// Created by ahueck on 20.10.20.
//

#include "CGFilter.h"

#include "CGInterface.h"

#include "llvm/IR/Intrinsics.h"

typeart::filter::CGFilter::CGFilter(const std::string& glob, bool CallFilterDeep, std::string file)
    : call_regex(util::glob2regex(glob)), CallFilterDeep(CallFilterDeep), trace(reason_trace) {
  if (!callGraph && !file.empty()) {
    LOG_FATAL("Resetting the CGInterface with JSON CG");
    // callGraph.reset(new JSONCG(JSONCG::getJSON(ClCGFile.getValue())));
    callGraph.reset(JSONCG::getJSON(file));
  } else {
    LOG_FATAL("CG File not found " << file);
  }
}
bool typeart::filter::CGFilter::filter(llvm::Value* in) {
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
              append_trace("CG maybe reaches ") << getName(c.getCalledFunction());
              continue;  // XXX This should be where we can change semantics
            } else {
              append_trace("Decl call (CG warn) ") << getName(c.getCalledFunction());
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

      const auto reached = do_cg(c.getCalledFunction());
      if (reached == CGInterface::ReachabilityResult::reaches) {
        append_trace("CG def. calls pattern ") << getName(c.getCalledFunction());
        return false;
      } else if (reached == CGInterface::ReachabilityResult::never_reaches) {
        append_trace("CG def. success ") << getName(c.getCalledFunction());
        continue;
      } else if (reached == CGInterface::ReachabilityResult::maybe_reaches) {
        append_trace("CG def. maybe reaches ") << getName(c.getCalledFunction());
        continue;  // XXX This should be where we can change semantics
      }

      // working_set_calls.push_back(c);
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
  return true;
}

bool typeart::filter::CGFilter::shouldContinue(CallSite c, llvm::Value* in) const {
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

std::string typeart::filter::CGFilter::getName(const llvm::Function* f) {
  auto name = f->getName();
  // FIXME figure out if we need to demangle, i.e., source is .c or .cpp
  const auto f_name = util::demangle(name);
  if (f_name != "") {
    name = f_name;
  }

  return name;
}
llvm::raw_string_ostream& typeart::filter::CGFilter::append_trace(std::string s) {
  trace << " | " << s;
  return trace;
}
std::string typeart::filter::CGFilter::reason() {
  return trace.str();
}
void typeart::filter::CGFilter::clear_trace() {
  reason_trace.clear();
}

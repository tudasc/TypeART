//
// Created by ahueck on 19.10.20.
//

#include "StandardFilter.h"

namespace typeart::filter {

StandardFilter::StandardFilter(const std::string& glob, bool CallFilterDeep)
    : call_regex(util::glob2regex(glob)), ClCallFilterDeep(CallFilterDeep) {
}

void StandardFilter::setMode(bool search_malloc) {
  malloc_mode = search_malloc;
}

void StandardFilter::setStartingFunction(llvm::Function* start) {
  start_f = start;
  depth   = 0;
}

bool StandardFilter::filter(Value* in) {
  if (in == nullptr) {
    LOG_DEBUG("Called with nullptr");
    return false;
  }

  if (depth == 15) {
    return false;
  }

  const auto match = [&](auto callee) -> bool {
    const auto name = StandardFilter::getName(callee);
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
        return false;  // Indirect calls might contain critical function calls.
      }

      const bool is_decl = callee->isDeclaration();
      // FIXME the MPI calls are all hitting this branch (obviously)
      if (is_decl) {
        LOG_DEBUG("Found call with declaration only. Call: " << util::dump(*c.getInstruction()));
        if (c.getIntrinsicID() == Intrinsic::not_intrinsic /*Intrinsic::ID::not_intrinsic*/) {
          if (ClCallFilterDeep && match(callee) && shouldContinue(c, in)) {
            continue;
          }
          return false;
        } else {
          LOG_DEBUG("Call is an intrinsic. Continue analyzing...")
          continue;
        }
      }

      if (match(callee)) {
        LOG_DEBUG("Found a call. Call: " << util::dump(*c.getInstruction()));
        if (ClCallFilterDeep && shouldContinue(c, in)) {
          continue;
        }
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
  return std::all_of(working_set_calls.begin(), working_set_calls.end(), [&](CallSite c) { return filter(c, in); });
}

bool StandardFilter::filter(CallSite& csite, Value* in) {
  const auto analyse_arg = [&](auto& csite, auto argNum) -> bool {
    Argument& the_arg = *(csite.getCalledFunction()->arg_begin() + argNum);
    LOG_DEBUG("Calling filter with inst of argument: " << util::dump(the_arg));
    const bool filter_arg = filter(&the_arg);
    LOG_DEBUG("Should filter? : " << filter_arg);
    return filter_arg;
  };

  LOG_DEBUG("Analyzing function call " << csite.getCalledFunction()->getName());

  if (csite.getCalledFunction() == start_f) {
    return true;
  }

  // this only works if we can correlate alloca with argument:
  const auto pos = std::find_if(csite.arg_begin(), csite.arg_end(),
                                [&in](const Use& arg_use) -> bool { return arg_use.get() == in; });
  // auto pos = csite.arg_end();
  if (pos != csite.arg_end()) {
    const auto argNum = std::distance(csite.arg_begin(), pos);
    LOG_DEBUG("Found exact position: " << argNum);
    return analyse_arg(csite, argNum);
  } else {
    LOG_DEBUG("Analyze all args, cannot correlate alloca with arg.");
    return std::all_of(csite.arg_begin(), csite.arg_end(), [&csite, &analyse_arg](const Use& arg_use) {
      auto argNum = csite.getArgumentNo(&arg_use);
      return analyse_arg(csite, argNum);
    });
  }

  return true;
}

bool StandardFilter::filter(Argument* arg) {
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

bool StandardFilter::shouldContinue(CallSite c, Value* in) const {
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
std::string StandardFilter::getName(const Function* f) {
  auto name = f->getName();
  // FIXME figure out if we need to demangle, i.e., source is .c or .cpp
  const auto f_name = util::demangle(name);
  if (f_name != "") {
    name = f_name;
  }

  return name;
}
}  // namespace typeart::filter

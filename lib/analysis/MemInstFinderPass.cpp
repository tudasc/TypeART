/*
 * MemInstFinderPass.cpp
 *
 *  Created on: Jun 3, 2018
 *      Author: ahueck
 */

#include "MemInstFinderPass.h"

#include "MemOpVisitor.h"
#include "support/Logger.h"
#include "support/TypeUtil.h"
#include "support/Util.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

#define DEBUG_TYPE "meminstanalysis"

namespace {
static RegisterPass<typeart::MemInstFinderPass> X("mem-inst-finder",                                // pass option
                                                  "Find heap and stack allocations in a program.",  // pass description
                                                  true,  // does not modify the CFG
                                                  true   // and it's an analysis
);
}  // namespace

static cl::opt<bool> ClMemInstFilter("alloca-filter", cl::desc("Filter alloca instructions."), cl::Hidden,
                                     cl::init(true));

static cl::opt<const char*> ClMemInstAllocaFilterStr("alloca-filter-str",
                                                     cl::desc("Filter alloca instructions based on string."),
                                                     cl::Hidden, cl::init("MPI_*"));

STATISTIC(NumDetectedAllocs, "Number of detected allocs");
STATISTIC(NumFilteredAllocs, "Number of filtered allocs");

namespace typeart {

namespace filter {

class CallFilter::FilterImpl {
  const std::string call_regex;

 public:
  FilterImpl(const std::string& glob) : call_regex(util::glob2regex(glob)) {
  }

  bool filter(Value* in) const {
    if (in == nullptr) {
      LOG_DEBUG("Called with nullptr");
      return false;
    }

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
        const auto callee = c.getCalledFunction();
        const bool indirect_call = callee == nullptr;

        if (indirect_call || callee->isDeclaration()) {
          LOG_DEBUG("Found an indirect call/only declaration, not filtering alloca. Call: " << *c.getInstruction());
          return false;  // Indirect calls might contain a critical function calls.
        }

        const auto name = FilterImpl::getName(callee);

        LOG_DEBUG("Found a call. Call: (" << *c.getInstruction() << ") Name: " << name);
        if (util::regex_matches(call_regex, name)) {
          LOG_DEBUG("Keeping alloca based on call name filter match");
          return false;
        }

        working_set_calls.push_back(c);
        // Caveat: below, we add users of the function call to the search even though it might be a
        // simple "sink" for the alloca we analyse
      }
      // cont. our search
      addToWork(val->users());
    }

    return std::all_of(working_set_calls.begin(), working_set_calls.end(), [&](CallSite c) { return filter(c, in); });
  }

 private:
  bool filter(CallSite& csite, Value* in) const {
    const auto analyse_arg = [&](auto& csite, auto argNum) -> bool {
      Argument& the_arg = *(csite.getCalledFunction()->arg_begin() + argNum);
      LOG_DEBUG("Calling filter with inst of argument: " << the_arg);
      const bool filter_arg = filter(&the_arg);
      LOG_DEBUG("Should filter? : " << filter_arg);
      return filter_arg;
    };

    LOG_DEBUG("Analyzing function call " << csite.getCalledFunction()->getName());

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

  bool filter(Argument* arg) const {
    for (auto* user : arg->users()) {
      LOG_DEBUG("Looking at arg user " << *user);
      // This code is for non mem2reg code (i.e., where the argument is stored to a local alloca):
      if (StoreInst* store = llvm::dyn_cast<StoreInst>(user)) {
        // if (auto* alloca = llvm::dyn_cast<AllocaInst>(store->getPointerOperand())) {
        //  LOG_DEBUG("Argument is a store inst and the operand is alloca");
        return filter(store->getPointerOperand());
        // }
      }
    }
    return filter(llvm::dyn_cast<Value>(arg));
  }

  static inline std::string getName(const Function* f) {
    auto name = f->getName();
    // FIXME figure out if we need to demangle, i.e., source is .c or .cpp
    const auto f_name = util::demangle(name);
    if (f_name != "") {
      name = f_name;
    }

    return name;
  }
};

CallFilter::CallFilter(const std::string& glob) : fImpl{std::make_unique<FilterImpl>(glob)} {
}

bool CallFilter::operator()(AllocaInst* in) {
  LOG_DEBUG("Analyzing value: " << *in);
  const auto filter_ = fImpl->filter(in);
  if (filter_) {
    LOG_DEBUG("Filtering value: " << *in << "\n");
    ++NumFilteredAllocs;
  } else {
    LOG_DEBUG("Keeping value: " << *in << "\n");
  }
  return filter_;
}

CallFilter& CallFilter::operator=(CallFilter&&) = default;

CallFilter::~CallFilter() = default;

}  // namespace filter

char MemInstFinderPass::ID = 0;

MemInstFinderPass::MemInstFinderPass()
    : llvm::FunctionPass(ID), mOpsCollector(), filter(ClMemInstAllocaFilterStr.getValue()) {
}

void MemInstFinderPass::getAnalysisUsage(llvm::AnalysisUsage& info) const {
  info.setPreservesAll();
}

bool MemInstFinderPass::runOnFunction(llvm::Function& f) {
  LOG_DEBUG("Running on function: " << f.getName())
  const auto checkAmbigiousMalloc = [](const MallocData& mallocData) {
    auto primaryBitcast = mallocData.primary;
    if (primaryBitcast) {
      const auto& bitcasts = mallocData.bitcasts;
      std::for_each(bitcasts.begin(), bitcasts.end(), [&](auto bitcastInst) {
        if (bitcastInst != primaryBitcast && (!typeart::util::type::isVoidPtr(bitcastInst->getDestTy()) &&
                                              primaryBitcast->getDestTy() != bitcastInst->getDestTy())) {
          // Second non-void* bitcast detected - semantics unclear
          LOG_WARNING("Encountered ambiguous pointer type in allocation: " << util::dump(*(mallocData.call)));
          LOG_WARNING("  Primary cast: " << util::dump(*primaryBitcast));
          LOG_WARNING("  Secondary cast: " << util::dump(*bitcastInst));
        }
      });
    }
  };

  mOpsCollector.clear();
  mOpsCollector.visit(f);

  if (ClMemInstFilter) {
    //    filter::CallFilter cfilter("MPI_*");
    auto& allocs = mOpsCollector.listAlloca;
    for (auto* alloc : allocs) {
      if (filter(alloc)) {
        allocs.erase(alloc);
      }
    }
    LOG_DEBUG("Allocas to instrument : " << util::dump(allocs));
  }

  for (const auto& mallocData : mOpsCollector.listMalloc) {
    checkAmbigiousMalloc(mallocData);
  }

  return false;
}

bool MemInstFinderPass::doFinalization(llvm::Module& m) {
  LOG_DEBUG("Found alloca count: " << NumDetectedAllocs);
  LOG_DEBUG("Filtered alloca count: " << NumFilteredAllocs);
  return false;
}

const llvm::SmallVector<MallocData, 8>& MemInstFinderPass::getFunctionMallocs() const {
  return mOpsCollector.listMalloc;
}

const llvm::SmallPtrSet<llvm::AllocaInst*, 8>& MemInstFinderPass::getFunctionAllocs() const {
  return mOpsCollector.listAlloca;
}

const llvm::SmallPtrSet<llvm::CallInst*, 8>& MemInstFinderPass::getFunctionFrees() const {
  return mOpsCollector.listFree;
}

}  // namespace typeart

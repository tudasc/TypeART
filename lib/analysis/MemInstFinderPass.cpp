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
  std::string call_regex;

 public:
  FilterImpl(const std::string& glob) : call_regex(util::glob2regex(glob)) {
  }

  bool filter(Argument* arg) {
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

  bool filter(Value* in) {
    if (in == nullptr) {
      LOG_DEBUG("Called with nullptr");
      return false;
    }

    llvm::SmallPtrSet<Value*, 16> visited_set;
    llvm::SmallVector<Value*, 16> working_set;
    llvm::SmallVector<CallSite, 16> working_set_calls;

    const auto addToWorkS = [&](auto v) {
      if (visited_set.find(v) == visited_set.end()) {
        working_set.push_back(v);
        visited_set.insert(v);
      }
    };

    const auto addToWork = [&addToWorkS](auto vals) {
      for (auto v : vals) {
        addToWorkS(v);
      }
    };

    const auto peek = [&working_set]() -> Value* {
      if (working_set.empty()) {
        return nullptr;
      }
      auto user_iter = working_set.end() - 1;
      working_set.erase(user_iter);
      return *user_iter;
    };

    // Seed working set with users of value (e.g., our AllocaInst)
    addToWork(in->users());

    // Search through all users of users .... (e.g., our AllocaInst)
    while (!working_set.empty()) {
      auto val = peek();

      // If we encounter a callsite, we want to analyze later, or quit in case we have a regex match
      CallSite c(val);
      if (c.isCall()) {
        const auto callee = c.getCalledFunction();
        const bool indirect_call = callee == nullptr;

        if (indirect_call) {
          LOG_DEBUG("Found an indirect call, not filtering alloca. Call: " << *c.getInstruction());
          return false;  // Indirect calls might contain a critical function.
        }

        auto name = callee->getName();
        auto f_name = util::demangle(name);
        if (f_name != "") {
          name = f_name;  // FIXME figure out if we need to demangle, i.e., source is .c or .cpp
        }

        LOG_DEBUG("Found a call. Call: " << *c.getInstruction() << " Name: " << name);
        if (callee->isDeclaration() || util::regex_matches(call_regex, name)) {
          LOG_DEBUG("Keeping alloca based on call. is_def: " << callee->isDeclaration());
          return false;
        }

        working_set_calls.push_back(c);
      }
      // Not a call, cont. our search
      addToWork(val->users());
    }

    if (working_set_calls.size() > 0) {
      // return false;  // we don't follow subfunctions yet
    }

    // Analyze the collected callsites (recursively)
    for (auto csite : working_set_calls) {
      LOG_DEBUG("Analyzing function call " << csite.getCalledFunction()->getName())
      for (Use& arg : csite.args()) {
        auto argNum = csite.getArgumentNo(&arg);
        auto arg_ = arg.get();
        LOG_DEBUG("Arg #" << argNum << " (" << *arg_ << ")");
        // If we encounter a call with **any** arg related to an (MPI) call we filter
        // TODO correlation between initial alloca and the arg num. For
        // now we are very inclusive and basically state, any subfunction, which our alloca is connected to should not
        // have a call_regex function name
        Argument& the_arg = *(csite.getCalledFunction()->arg_begin() + argNum);
        // for (auto arg_user : the_arg.users()) {
        LOG_DEBUG("Calling filter with inst ("
                  << ") of argument: " << the_arg);
        const bool filter_arg = filter(&the_arg);
        LOG_DEBUG("Should filter? : " << filter_arg);
        if (!filter_arg) {
          return false;
        }
        //}
      }
    }

    return true;
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
    LOG_DEBUG("Result : " << util::dump(allocs));
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

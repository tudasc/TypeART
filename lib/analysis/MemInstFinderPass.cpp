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

STATISTIC(NumDetectedAllocs, "Number of detected allocs");
STATISTIC(NumFilteredAllocs, "Number of filtered allocs");

namespace typeart {

struct CallFilter {
  std::string call_regex;

  explicit CallFilter(const std::string& glob) : call_regex(util::glob2regex(glob)) {
  }

  bool operator()(AllocaInst* in) {
    const auto filter_ = filter(in);
    if (filter_) {
      ++NumFilteredAllocs;
    }
    return filter_;
  }

 private:
  bool filter(Value* in) {
    if (in == nullptr) {
      return false;
    }

    LOG_DEBUG("Filtering value: " << in);

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
    for (auto user : in->users()) {
      working_set.push_back(user);
    }

    // Search through all users of users .... (e.g., our AllocaInst)
    while (!working_set.empty()) {
      auto val = peek();

      // If we encounter a callsite, we want to analyze later, or quit in case we have a regex match
      CallSite c(val);
      if (c.isCall()) {
        const bool indirect_call = c.getCalledFunction() == nullptr;

        if (indirect_call) {
          LOG_DEBUG("Found an indirect call, not filtering alloca. Call: " << c.getInstruction());
          return false;  // Indirect calls might contain a critical function.
        }

        LOG_DEBUG("Found a call as a user of alloca. Call: " << c.getInstruction());
        auto f_name = util::demangle(c.getCalledFunction()->getName());
        if (util::regex_matches(call_regex, f_name)) {
          LOG_DEBUG("Filtering alloca based on call!");
          return false;
        }
        working_set_calls.push_back(c);
      }
      // Not a call, cont. our search
      addToWork(val->users());
    }

    // Analyze the collected callsites (recursively)
    for (auto csite : working_set_calls) {
      for (Use& arg : csite.args()) {
        // If we encounter a call with any arg related to an (MPI) call we filter
        // TODO correlation between initial alloca and the arg num. For
        // now we are very inclusive and basically state, any subfunction, which our alloca is connected to should not
        // have a call_regex function name
        const bool filter_arg = filter(arg.get());
        if (!filter_arg) {
          return false;
        }
      }
    }

    return true;
  }
};

char MemInstFinderPass::ID = 0;

MemInstFinderPass::MemInstFinderPass() : llvm::FunctionPass(ID) {
}

void MemInstFinderPass::getAnalysisUsage(llvm::AnalysisUsage& info) const {
  info.setPreservesAll();
}

bool MemInstFinderPass::runOnFunction(llvm::Function& f) {
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
    CallFilter cfilter("MPI_*");
    auto& allocs = mOpsCollector.listAlloca;
    for (auto* alloc : allocs) {
      if (cfilter(alloc)) {
        allocs.erase(alloc);
      }
    }
  }

  for (const auto& mallocData : mOpsCollector.listMalloc) {
    checkAmbigiousMalloc(mallocData);
  }

  return false;
}

bool MemInstFinderPass::doFinalization(llvm::Module&) {
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

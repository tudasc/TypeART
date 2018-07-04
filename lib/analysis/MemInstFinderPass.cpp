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

using namespace llvm;

#define DEBUG_TYPE "meminstanalysis"

namespace {
static RegisterPass<typeart::MemInstFinderPass> X("mem-inst-finder",                                // pass option
                                                  "Find heap and stack allocations in a program.",  // pass description
                                                  true,  // does not modify the CFG
                                                  true   // and it's an analysis
);
}  // namespace

static cl::opt<bool> ClFilterMallocAllocPair("mem-inst-malloc-store-filter",
                                             cl::desc("Filter allocs that get a store from a heap alloc."), cl::Hidden,
                                             cl::init(false));

namespace typeart {

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

  if (ClFilterMallocAllocPair) {
    auto& alist = mOpsCollector.listAlloca;
    auto& mlist = mOpsCollector.listMalloc;

    const auto filterMallocAllocPairing = [&mlist](const auto alloc) {
      // Only look for the direct users of the alloc:
      // TODO is a deeper analysis required?
      for (auto inst : alloc->users()) {
        if (StoreInst* store = dyn_cast<StoreInst>(inst)) {
          const auto source = store->getValueOperand();
          if (isa<BitCastInst>(source)) {
            for (auto& mdata : mlist) {
              // is it a bitcast we already collected? if yes, we can filter the alloc
              return std::any_of(mdata.bitcasts.begin(), mdata.bitcasts.end(),
                                 [&source](const auto bcast) { return bcast == source; });
            }
          } else if (isa<CallInst>(source)) {
            return std::any_of(mlist.begin(), mlist.end(),
                               [&source](const auto& mdata) { return mdata.call == source; });
          }
        }
      }
      return false;
    };

    for (auto alloc : alist) {
      LOG_DEBUG("Filtering allocs (used to store a heap alloc pointer!) in function: " << f.getName());
      if (filterMallocAllocPairing(alloc)) {
        LOG_DEBUG("Filtering alloc: " << util::dump(*alloc));
        alist.erase(alloc);
      }
    }
  }

  for (const auto& mallocData : mOpsCollector.listMalloc) {
    checkAmbigiousMalloc(mallocData);
  }

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

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

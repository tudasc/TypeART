/*
 * MemInstFinderPass.h
 *
 *  Created on: Jun 3, 2018
 *      Author: ahueck
 */

#ifndef LIB_ANALYSIS_MEMINSTFINDERPASS_H_
#define LIB_ANALYSIS_MEMINSTFINDERPASS_H_

#include "MemOpVisitor.h"
#include "llvm/Pass.h"

namespace llvm {
class Function;
}  // namespace llvm

namespace typeart {

class MemInstFinderPass : public llvm::FunctionPass {
 private:
  MemOpVisitor mOpsCollector;

 public:
  static char ID;
  MemInstFinderPass();
  bool runOnFunction(llvm::Function&) override;
  void getAnalysisUsage(llvm::AnalysisUsage&) const override;
  bool doFinalization(llvm::Module&) override;
  const llvm::SmallVector<MallocData, 8>& getFunctionMallocs() const;
  const llvm::SmallPtrSet<llvm::AllocaInst*, 8>& getFunctionAllocs() const;
  const llvm::SmallPtrSet<llvm::CallInst*, 8>& getFunctionFrees() const;
};

}  // namespace typeart

#endif /* LIB_ANALYSIS_MEMINSTFINDERPASS_H_ */

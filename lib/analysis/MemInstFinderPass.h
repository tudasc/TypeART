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

#include <memory>

namespace llvm {
class Function;
class AllocaInst;
class CallSite;
}  // namespace llvm

namespace typeart {

namespace filter {

class CallFilter {
  class FilterImpl;
  std::unique_ptr<FilterImpl> fImpl;

 public:
  explicit CallFilter(const std::string& glob);
  CallFilter(const CallFilter&) = delete;
  CallFilter(CallFilter&&) = default;
  bool operator()(llvm::AllocaInst* in);
  bool operator()(llvm::CallSite in);
  CallFilter& operator=(CallFilter&&) noexcept;
  CallFilter& operator=(const CallFilter&) = delete;
  virtual ~CallFilter();
};

}  // namespace filter

class MemInstFinderPass : public llvm::FunctionPass {
 private:
  MemOpVisitor mOpsCollector;
  filter::CallFilter filter;

 public:
  static char ID;
  MemInstFinderPass();
  bool runOnFunction(llvm::Function&) override;
  void getAnalysisUsage(llvm::AnalysisUsage&) const override;
  bool doFinalization(llvm::Module&) override;
  const llvm::SmallVector<MallocData, 8>& getFunctionMallocs() const;
  const llvm::SmallVector<AllocaData, 8>& getFunctionAllocs() const;
  const llvm::SmallPtrSet<llvm::CallInst*, 8>& getFunctionFrees() const;
};

}  // namespace typeart

#endif /* LIB_ANALYSIS_MEMINSTFINDERPASS_H_ */

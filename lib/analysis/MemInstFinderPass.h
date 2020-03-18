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
class Module;
class Function;
class AllocaInst;
class CallSite;
class GlobalValue;
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
  bool operator()(llvm::AllocaInst*);
  bool operator()(llvm::GlobalValue*);
  CallFilter& operator=(CallFilter&&) noexcept;
  CallFilter& operator=(const CallFilter&) = delete;
  virtual ~CallFilter();
};

}  // namespace filter

struct FunctionData {
  llvm::SmallVector<MallocData, 8> listMalloc;
  llvm::SmallPtrSet<llvm::CallInst*, 8> listFree;
  llvm::SmallVector<AllocaData, 8> listAlloca;
  llvm::SmallVector<AssertData, 8> listAssert;
};

class MemInstFinderPass : public llvm::ModulePass {
 private:
  finder::MemOpVisitor mOpsCollector;
  filter::CallFilter filter;
  llvm::DenseMap<llvm::Function*, FunctionData> functionMap;

 public:
  static char ID;

  MemInstFinderPass();
  bool runOnModule(llvm::Module&) override;
  bool runOnFunc(llvm::Function&);
  void getAnalysisUsage(llvm::AnalysisUsage&) const override;
  bool doFinalization(llvm::Module&) override;
  bool hasFunctionData(llvm::Function*) const;
  const FunctionData& getFunctionData(llvm::Function*) const;
  const llvm::SmallVector<llvm::GlobalVariable*, 8>& getModuleGlobals() const;

 private:
  void printStats(llvm::raw_ostream&);
};

}  // namespace typeart

#endif /* LIB_ANALYSIS_MEMINSTFINDERPASS_H_ */

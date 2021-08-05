// TypeART library
//
// Copyright (c) 2017-2021 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#ifndef LIB_ANALYSIS_MEMINSTFINDERPASS_H_
#define LIB_ANALYSIS_MEMINSTFINDERPASS_H_

#include "MemOpData.h"
#include "MemOpVisitor.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Pass.h"

#include <memory>
#include <string>

namespace llvm {
class Module;
class Function;
class AllocaInst;
class CallSite;
class GlobalValue;
class AnalysisUsage;
class raw_ostream;
}  // namespace llvm

namespace typeart {

namespace filter {
class Filter;

class CallFilter {
  std::unique_ptr<Filter> fImpl;

 public:
  explicit CallFilter(const std::string& glob);
  CallFilter(const CallFilter&) = delete;
  CallFilter(CallFilter&&)      = default;
  bool operator()(llvm::AllocaInst*);
  bool operator()(llvm::GlobalValue*);
  CallFilter& operator=(CallFilter&&) noexcept;
  CallFilter& operator=(const CallFilter&) = delete;
  virtual ~CallFilter();
};

}  // namespace filter

struct FunctionData {
  MallocDataList mallocs;
  FreeDataList frees;
  AllocaDataList allocas;
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
  const GlobalDataList& getModuleGlobals() const;

 private:
  void printStats(llvm::raw_ostream&);
};

}  // namespace typeart

#endif /* LIB_ANALYSIS_MEMINSTFINDERPASS_H_ */

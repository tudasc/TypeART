// TypeART library
//
// Copyright (c) 2017-2022 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#ifndef _LIB_MUSTSUPPORTPASS_H
#define _LIB_MUSTSUPPORTPASS_H

#include "instrumentation/Instrumentation.h"
#include "instrumentation/InstrumentationHelper.h"
#include "instrumentation/TypeARTFunctions.h"

#include "llvm/Pass.h"

#include <memory>
#include <string>

namespace llvm {
class Module;
class Function;
class AnalysisUsage;
class Value;
class raw_ostream;
}  // namespace llvm

namespace typeart {
class TypeGenerator;

namespace analysis {
class MemInstFinder;
}  // namespace analysis

}  // namespace typeart

namespace typeart::pass {

class TypeArtPass : public llvm::ModulePass {
 private:
  const std::string default_types_file{"types.yaml"};

  struct TypeArtFunc {
    const std::string name;
    llvm::Value* f{nullptr};
  };

  TypeArtFunc typeart_alloc{"__typeart_alloc"};
  TypeArtFunc typeart_alloc_global{"__typeart_alloc_global"};
  TypeArtFunc typeart_alloc_stack{"__typeart_alloc_stack"};
  TypeArtFunc typeart_free{"__typeart_free"};
  TypeArtFunc typeart_leave_scope{"__typeart_leave_scope"};

  TypeArtFunc typeart_alloc_omp        = typeart_alloc;
  TypeArtFunc typeart_alloc_stacks_omp = typeart_alloc_stack;
  TypeArtFunc typeart_free_omp         = typeart_free;
  TypeArtFunc typeart_leave_scope_omp  = typeart_leave_scope;

  std::unique_ptr<analysis::MemInstFinder> meminst_finder;
  std::unique_ptr<TypeGenerator> typeManager;
  InstrumentationHelper instrumentation_helper;
  TAFunctions functions;
  std::unique_ptr<InstrumentationContext> instrumentation_context;

 public:
  static char ID;  // used to identify pass

  TypeArtPass();
  bool doInitialization(llvm::Module&) override;
  bool runOnModule(llvm::Module&) override;
  bool runOnFunc(llvm::Function&);
  bool doFinalization(llvm::Module&) override;
  void getAnalysisUsage(llvm::AnalysisUsage&) const override;

 private:
  void declareInstrumentationFunctions(llvm::Module&);
  void printStats(llvm::raw_ostream&);
};

}  // namespace typeart::pass

#endif

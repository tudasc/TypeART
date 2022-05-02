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

#include "TypeARTPass.h"

#include "analysis/MemInstFinder.h"
#include "instrumentation/MemOpArgCollector.h"
#include "instrumentation/MemOpInstrumentation.h"
#include "instrumentation/TypeARTFunctions.h"
#include "support/Logger.h"
#include "support/Table.h"
#include "typegen/TypeGenerator.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstddef>
#include <sstream>
#include <string>
#include <utility>

namespace llvm {
class BasicBlock;
}  // namespace llvm

using namespace llvm;

#define DEBUG_TYPE "typeart"

static llvm::RegisterPass<typeart::pass::TypeArtPass> msp("typeart", "TypeArt type instrumentation sanitizer", false,
                                                          false);

static cl::OptionCategory typeart_category("TypeART instrumentation pass", "These control the instrumentation.");

static cl::opt<std::string> cl_typeart_type_file("typeart-types", cl::desc("Location of the generated type file."),
                                                 cl::cat(typeart_category));

static cl::opt<bool> cl_typeart_stats("typeart-stats", cl::desc("Show statistics for TypeArt type pass."), cl::Hidden,
                                      cl::init(false), cl::cat(typeart_category));

static cl::opt<bool> cl_typeart_instrument_heap("typeart-heap",
                                                cl::desc("Instrument heap allocation/free instructions."),
                                                cl::init(true), cl::cat(typeart_category));

static cl::opt<bool> cl_typeart_instrument_global("typeart-global", cl::desc("Instrument global allocations."),
                                                  cl::init(false), cl::cat(typeart_category));

static cl::opt<bool> cl_typeart_instrument_stack(
    "typeart-stack", cl::desc("Instrument stack (alloca) allocations. Turns on global instrumentation."),
    cl::init(false), cl::cat(typeart_category), cl::callback([](const bool& opt) {
      if (opt) {
        ::cl_typeart_instrument_global = true;
      }
    }));

static cl::opt<bool> cl_typeart_instrument_stack_lifetime(
    "typeart-stack-lifetime", cl::desc("Instrument lifetime.start intrinsic instead of alloca."), cl::init(true),
    cl::cat(typeart_category));

static cl::OptionCategory typeart_meminstfinder_category(
    "TypeART memory instruction finder", "These options control which memory instructions are collected/filtered.");

static cl::opt<bool> cl_typeart_filter_stack_non_array("typeart-stack-array-only",
                                                       cl::desc("Only find stack (alloca) instructions of arrays."),
                                                       cl::Hidden, cl::init(false),
                                                       cl::cat(typeart_meminstfinder_category));

static cl::opt<bool> cl_typeart_filter_heap_alloc(
    "typeart-malloc-store-filter", cl::desc("Filter alloca instructions that have a store from a heap allocation."),
    cl::Hidden, cl::init(false), cl::cat(typeart_meminstfinder_category));

static cl::opt<bool> cl_typeart_filter_global("typeart-filter-globals", cl::desc("Filter globals of a module."),
                                              cl::Hidden, cl::init(true), cl::cat(typeart_meminstfinder_category));

static cl::opt<bool> cl_typeart_call_filter(
    "typeart-call-filter",
    cl::desc("Filter (stack/global) alloca instructions that are passed to specific function calls."), cl::Hidden,
    cl::init(false), cl::cat(typeart_meminstfinder_category));

static cl::opt<typeart::analysis::FilterImplementation> cl_typeart_call_filter_implementation(
    "typeart-call-filter-impl", cl::desc("Select the call filter implementation."),
    cl::values(clEnumValN(typeart::analysis::FilterImplementation::none, "none", "No filter"),
               clEnumValN(typeart::analysis::FilterImplementation::standard, "std",
                          "Standard forward filter (default)"),
               clEnumValN(typeart::analysis::FilterImplementation::cg, "cg", "Call-graph-based filter")),
    cl::Hidden, cl::init(typeart::analysis::FilterImplementation::standard), cl::cat(typeart_meminstfinder_category));

static cl::opt<std::string> cl_typeart_call_filter_glob(
    "typeart-call-filter-str", cl::desc("Filter allocas based on the function name (glob) <string>."), cl::Hidden,
    cl::init("*MPI_*"), cl::cat(typeart_meminstfinder_category));

static cl::opt<std::string> cl_typeart_call_filter_glob_deep(
    "typeart-call-filter-deep-str",
    cl::desc("Filter allocas based on specific API, i.e., value passed as void* are correlated when string matched and "
             "possibly kept."),
    cl::Hidden, cl::init("MPI_*"), cl::cat(typeart_meminstfinder_category));

static cl::opt<std::string> cl_typeart_call_filter_cg_file("typeart-call-filter-cg-file",
                                                           cl::desc("Location of call-graph file to use."), cl::Hidden,
                                                           cl::init(""), cl::cat(typeart_meminstfinder_category));

static cl::opt<bool> cl_typeart_filter_pointer_alloca("typeart-filter-pointer-alloca",
                                                      cl::desc("Filter allocas of pointer types."), cl::Hidden,
                                                      cl::init(true), cl::cat(typeart_meminstfinder_category));

ALWAYS_ENABLED_STATISTIC(NumInstrumentedMallocs, "Number of instrumented mallocs");
ALWAYS_ENABLED_STATISTIC(NumInstrumentedFrees, "Number of instrumented frees");
ALWAYS_ENABLED_STATISTIC(NumInstrumentedAlloca, "Number of instrumented (stack) allocas");
ALWAYS_ENABLED_STATISTIC(NumInstrumentedGlobal, "Number of instrumented globals");

namespace typeart::pass {

// Used by LLVM pass manager to identify passes in memory
char TypeArtPass::ID = 0;

TypeArtPass::TypeArtPass() : llvm::ModulePass(ID) {
  analysis::MemInstFinderConfig conf{cl_typeart_instrument_heap,                                                   //
                                     cl_typeart_instrument_stack,                                                  //
                                     cl_typeart_instrument_global,                                                 //
                                     analysis::MemInstFinderConfig::Filter{cl_typeart_filter_stack_non_array,      //
                                                                           cl_typeart_filter_heap_alloc,           //
                                                                           cl_typeart_filter_global,               //
                                                                           cl_typeart_call_filter,                 //
                                                                           cl_typeart_filter_pointer_alloca,       //
                                                                           cl_typeart_call_filter_implementation,  //
                                                                           cl_typeart_call_filter_glob,            //
                                                                           cl_typeart_call_filter_glob_deep,       //
                                                                           cl_typeart_call_filter_cg_file}};
  meminst_finder = analysis::create_finder(conf);

  EnableStatistics(false);
}

void TypeArtPass::getAnalysisUsage(llvm::AnalysisUsage& info) const {
}

bool TypeArtPass::doInitialization(Module& m) {
  const auto types_file = [&]() -> std::string {
    if (!cl_typeart_type_file.empty()) {
      LOG_DEBUG("Using cl::opt for types file " << cl_typeart_type_file.getValue())
      return cl_typeart_type_file.getValue();
    }
    const char* type_file = std::getenv("TYPEART_TYPE_FILE");
    if (type_file != nullptr) {
      LOG_DEBUG("Using env var for types file " << type_file)
      return std::string{type_file};
    }
    LOG_DEBUG("Loading default types file " << default_types_file)
    return default_types_file;
  }();

  typeManager = make_typegen(types_file);

  LOG_DEBUG("Propagating type infos.");
  const auto [loaded, error] = typeManager->load();
  if (loaded) {
    LOG_DEBUG("Existing type configuration successfully loaded from " << cl_typeart_type_file.getValue());
  } else {
    LOG_DEBUG("No valid existing type configuration found: " << cl_typeart_type_file.getValue()
                                                             << ". Reason: " << error.message());
  }

  instrumentation_helper.setModule(m);

  auto arg_collector = std::make_unique<MemOpArgCollector>(typeManager.get(), instrumentation_helper);
  auto mem_instrument =
      std::make_unique<MemOpInstrumentation>(functions, instrumentation_helper, cl_typeart_instrument_stack_lifetime);
  instrumentation_context =
      std::make_unique<InstrumentationContext>(std::move(arg_collector), std::move(mem_instrument));

  return true;
}

bool TypeArtPass::runOnModule(Module& m) {
  meminst_finder->runOnModule(m);

  bool instrumented_global{false};
  if (cl_typeart_instrument_global) {
    declareInstrumentationFunctions(m);

    const auto& globalsList = meminst_finder->getModuleGlobals();
    if (!globalsList.empty()) {
      const auto global_count = instrumentation_context->handleGlobal(globalsList);
      NumInstrumentedGlobal += global_count;
      instrumented_global = global_count > 0;
    }
  }

  const auto instrumented_function = llvm::count_if(m.functions(), [&](auto& f) { return runOnFunc(f); }) > 0;
  return instrumented_function || instrumented_global;
}

bool TypeArtPass::runOnFunc(Function& f) {
  using namespace typeart;

  if (f.isDeclaration() || f.getName().startswith("__typeart")) {
    return false;
  }

  if (!meminst_finder->hasFunctionData(f)) {
    LOG_WARNING("No allocation data could be retrieved for function: " << f.getName());
    return false;
  }

  LOG_DEBUG("Running on function: " << f.getName())

  // FIXME this is required when "PassManagerBuilder::EP_OptimizerLast" is used as the function (constant) pointer are
  // nullpointer/invalidated
  declareInstrumentationFunctions(*f.getParent());

  bool mod{false};
  //  auto& c = f.getContext();
  DataLayout dl(f.getParent());

  llvm::SmallDenseMap<BasicBlock*, size_t> allocCounts;

  const auto& fData   = meminst_finder->getFunctionData(f);
  const auto& mallocs = fData.mallocs;
  const auto& allocas = fData.allocas;
  const auto& frees   = fData.frees;

  if (cl_typeart_instrument_heap) {
    // instrument collected calls of bb:
    const auto heap_count = instrumentation_context->handleHeap(mallocs);
    const auto free_count = instrumentation_context->handleFree(frees);

    NumInstrumentedMallocs += heap_count;
    NumInstrumentedFrees += free_count;

    mod |= heap_count > 0 || free_count > 0;
  }

  if (cl_typeart_instrument_stack) {
    const auto stack_count = instrumentation_context->handleStack(allocas);
    NumInstrumentedAlloca += stack_count;
    mod |= stack_count > 0;
  }

  return mod;
}  // namespace pass

bool TypeArtPass::doFinalization(Module&) {
  /*
   * Persist the accumulated type definition information for this module.
   */
  LOG_DEBUG("Writing type file to " << cl_typeart_type_file.getValue());

  const auto [stored, error] = typeManager->store();
  if (stored) {
    LOG_DEBUG("Success!");
  } else {
    LOG_FATAL("Failed writing type config to " << cl_typeart_type_file.getValue() << ". Reason: " << error.message());
  }
  if (cl_typeart_stats) {
    auto& out = llvm::errs();
    printStats(out);
  }
  return false;
}

void TypeArtPass::declareInstrumentationFunctions(Module& m) {
  // Remove this return if problems come up during compilation
  if (typeart_alloc_global.f != nullptr && typeart_alloc_stack.f != nullptr && typeart_alloc.f != nullptr &&
      typeart_free.f != nullptr && typeart_leave_scope.f != nullptr) {
    return;
  }

  TAFunctionDeclarator decl(m, instrumentation_helper, functions);

  auto alloc_arg_types      = instrumentation_helper.make_parameters(IType::ptr, IType::type_id, IType::extent);
  auto free_arg_types       = instrumentation_helper.make_parameters(IType::ptr);
  auto leavescope_arg_types = instrumentation_helper.make_parameters(IType::stack_count);

  typeart_alloc.f        = decl.make_function(IFunc::heap, typeart_alloc.name, alloc_arg_types);
  typeart_alloc_stack.f  = decl.make_function(IFunc::stack, typeart_alloc_stack.name, alloc_arg_types);
  typeart_alloc_global.f = decl.make_function(IFunc::global, typeart_alloc_global.name, alloc_arg_types);
  typeart_free.f         = decl.make_function(IFunc::free, typeart_free.name, free_arg_types);
  typeart_leave_scope.f  = decl.make_function(IFunc::scope, typeart_leave_scope.name, leavescope_arg_types);

  typeart_alloc_omp.f = decl.make_function(IFunc::heap_omp, typeart_alloc_omp.name, alloc_arg_types, true);
  typeart_alloc_stacks_omp.f =
      decl.make_function(IFunc::stack_omp, typeart_alloc_stacks_omp.name, alloc_arg_types, true);
  typeart_free_omp.f = decl.make_function(IFunc::free_omp, typeart_free_omp.name, free_arg_types, true);
  typeart_leave_scope_omp.f =
      decl.make_function(IFunc::scope_omp, typeart_leave_scope_omp.name, leavescope_arg_types, true);
}

void TypeArtPass::printStats(llvm::raw_ostream& out) {
  meminst_finder->printStats(out);

  const auto get_ta_mode = [&]() {
    const bool heap  = cl_typeart_instrument_heap.getValue();
    const bool stack = cl_typeart_instrument_stack.getValue();

    if (heap) {
      if (stack) {
        return " [Heap & Stack]";
      }
      return " [Heap]";
    }

    if (stack) {
      return " [Stack]";
    }

    llvm_unreachable("Did not find heap or stack, or combination thereof!");
  };

  Table stats("TypeArtPass");
  stats.wrap_header = true;
  stats.title += get_ta_mode();
  stats.put(Row::make("Malloc", NumInstrumentedMallocs.getValue()));
  stats.put(Row::make("Free", NumInstrumentedFrees.getValue()));
  stats.put(Row::make("Alloca", NumInstrumentedAlloca.getValue()));
  stats.put(Row::make("Global", NumInstrumentedGlobal.getValue()));

  std::ostringstream stream;
  stats.print(stream);
  out << stream.str();
}

}  // namespace typeart::pass

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

static void registerClangPass(const llvm::PassManagerBuilder&, llvm::legacy::PassManagerBase& PM) {
  PM.add(new typeart::pass::TypeArtPass());
}
static RegisterStandardPasses RegisterClangPass(PassManagerBuilder::EP_OptimizerLast, registerClangPass);

#include "TypeARTPass.h"

#include "analysis/MemInstFinderPass.h"
#include "instrumentation/MemOpArgCollector.h"
#include "instrumentation/MemOpInstrumentation.h"
#include "instrumentation/TypeARTFunctions.h"
#include "support/Logger.h"
#include "support/Table.h"
#include "typelib/TypeDB.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/PassAnalysisSupport.h"
#include "llvm/PassSupport.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstddef>
#include <string>
#include <utility>

namespace llvm {
class BasicBlock;
}  // namespace llvm

using namespace llvm;

#define DEBUG_TYPE "typeart"

namespace {
static llvm::RegisterPass<typeart::pass::TypeArtPass> msp("typeart", "TypeArt type information", false, false);
}  // namespace

static cl::opt<bool> ClTypeArtStats("typeart-stats", cl::desc("Show statistics for TypeArt type pass."), cl::Hidden,
                                    cl::init(false));
extern cl::opt<bool> ClIgnoreHeap;
extern cl::opt<bool> ClTypeArtAlloca;

static cl::opt<std::string> ClTypeFile("typeart-outfile", cl::desc("Location of the generated type file."), cl::Hidden,
                                       cl::init("types.yaml"));

STATISTIC(NumInstrumentedMallocs, "Number of instrumented mallocs");
STATISTIC(NumInstrumentedFrees, "Number of instrumented frees");
STATISTIC(NumInstrumentedAlloca, "Number of instrumented (stack) allocas");
STATISTIC(NumInstrumentedGlobal, "Number of instrumented globals");

namespace typeart::pass {

// Used by LLVM pass manager to identify passes in memory
char TypeArtPass::ID = 0;

// std::unique_ptr<TypeMapping> TypeArtPass::typeMapping = std::make_unique<SimpleTypeMapping>();

TypeArtPass::TypeArtPass() : llvm::ModulePass(ID), typeManager(ClTypeFile.getValue()) {
  assert(!ClTypeFile.empty() && "Default type file not set");
  EnableStatistics();
}

void TypeArtPass::getAnalysisUsage(llvm::AnalysisUsage& info) const {
  info.addRequired<typeart::MemInstFinderPass>();
}

bool TypeArtPass::doInitialization(Module& m) {
  instrumentation_helper.setModule(m);

  LOG_DEBUG("Propagating type infos.");
  if (typeManager.load()) {
    LOG_DEBUG("Existing type configuration successfully loaded from " << ClTypeFile.getValue());
  } else {
    LOG_DEBUG("No valid existing type configuration found: " << ClTypeFile.getValue());
  }

  auto arg_collector  = std::make_unique<MemOpArgCollector>(typeManager, instrumentation_helper);
  auto mem_instrument = std::make_unique<MemOpInstrumentation>(functions, instrumentation_helper);
  instrumentation_context =
      std::make_unique<InstrumentationContext>(std::move(arg_collector), std::move(mem_instrument));

  return true;
}

bool TypeArtPass::runOnModule(Module& m) {
  bool instrumented_global{false};
  if (ClTypeArtAlloca) {
    declareInstrumentationFunctions(m);

    const auto& globalsList = getAnalysis<MemInstFinderPass>().getModuleGlobals();
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

  if (!getAnalysis<MemInstFinderPass>().hasFunctionData(&f)) {
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

  const auto& fData   = getAnalysis<MemInstFinderPass>().getFunctionData(&f);
  const auto& mallocs = fData.mallocs;
  const auto& allocas = fData.allocas;
  const auto& frees   = fData.frees;

  if (!ClIgnoreHeap) {
    // instrument collected calls of bb:
    const auto heap_count = instrumentation_context->handleHeap(mallocs);
    const auto free_count = instrumentation_context->handleFree(frees);

    NumInstrumentedMallocs += heap_count;
    NumInstrumentedFrees += free_count;

    mod |= heap_count > 0 || free_count > 0;
  }
  if (ClTypeArtAlloca) {
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
  LOG_DEBUG("Writing type file to " << ClTypeFile.getValue());

  if (typeManager.store()) {
    LOG_DEBUG("Success!");
  } else {
    LOG_FATAL("Failed writing type config to " << ClTypeFile.getValue());
  }
  if (ClTypeArtStats && AreStatisticsEnabled()) {
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
  const auto get_ta_mode = [&]() {
    const bool heap  = !ClIgnoreHeap.getValue();
    const bool stack = ClTypeArtAlloca.getValue();
    if (heap) {
      if (stack) {
        return " [Heap & Stack]";
      }
      return " [Heap]";
    } else if (stack) {
      return " [Stack]";
    }
    return " [Unknown]";
  };

  Table stats("TypeArtPass");
  stats.wrap_header = true;
  stats.title += get_ta_mode();
  stats.put(Row::make("Malloc", NumInstrumentedMallocs.getValue()));
  stats.put(Row::make("Free", NumInstrumentedFrees.getValue()));
  stats.put(Row::make("Alloca", NumInstrumentedAlloca.getValue()));
  stats.put(Row::make("Global", NumInstrumentedGlobal.getValue()));

  stats.print(out);
}

}  // namespace typeart::pass

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

static void registerClangPass(const llvm::PassManagerBuilder&, llvm::legacy::PassManagerBase& PM) {
  PM.add(new typeart::pass::TypeArtPass());
}
static RegisterStandardPasses RegisterClangPass(PassManagerBuilder::EP_OptimizerLast, registerClangPass);

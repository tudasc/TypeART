#include "TypeARTPass.h"

#include "RuntimeInterface.h"
#include "TypeIO.h"
#include "analysis/MemInstFinderPass.h"
#include "instrumentation/MemOpArgCollector.h"
#include "instrumentation/MemOpInstrumentation.h"
#include "instrumentation/TypeARTFunctions.h"
#include "support/Logger.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Transforms/Utils/CtorUtils.h"

#include <string>

using namespace llvm;

#define DEBUG_TYPE "typeart"

namespace {
static llvm::RegisterPass<typeart::pass::TypeArtPass> msp("typeart", "TypeArt type information", false, false);
}  // namespace

static cl::opt<bool> ClTypeArtStats("typeart-stats", cl::desc("Show statistics for TypeArt type pass."), cl::Hidden,
                                    cl::init(false));
static cl::opt<bool> ClIgnoreHeap("typeart-no-heap", cl::desc("Ignore heap allocation/free instruction."), cl::Hidden,
                                  cl::init(false));
static cl::opt<bool> ClTypeArtAlloca("typeart-alloca", cl::desc("Track alloca instructions."), cl::Hidden,
                                     cl::init(false));

static cl::opt<std::string> ClTypeFile("typeart-outfile", cl::desc("Location of the generated type file."), cl::Hidden,
                                       cl::init("types.yaml"));

STATISTIC(NumInstrumentedMallocs, "Number of instrumented mallocs");
STATISTIC(NumInstrumentedFrees, "Number of instrumented frees");
STATISTIC(NumInstrumentedAlloca, "Number of instrumented (stack) allocas");
STATISTIC(NumInstrumentedGlobal, "Number of instrumented globals");

namespace typeart {
namespace pass {

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

  arg_collector  = std::make_unique<MemOpArgCollector>(typeManager, instrumentation_helper);
  mem_instrument = std::make_unique<MemOpInstrumentation>(functions, instrumentation_helper);

  return true;
}

bool TypeArtPass::runOnModule(Module& m) {
  bool instrumented_global{false};
  if (ClIgnoreHeap) {
    declareInstrumentationFunctions(m);

    const auto& globalsList = getAnalysis<MemInstFinderPass>().getModuleGlobals();
    if (!globalsList.empty()) {
      auto global_args = arg_collector->collectGlobal(globalsList);

      const auto global_count = mem_instrument->instrumentGlobal(global_args);
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
    auto heap_args = arg_collector->collectHeap(mallocs);
    auto free_args = arg_collector->collectFree(frees);

    const auto heap_count = mem_instrument->instrumentHeap(heap_args);
    const auto free_count = mem_instrument->instrumentFree(free_args);

    NumInstrumentedMallocs += heap_count;
    NumInstrumentedFrees += free_count;

    mod |= heap_count > 0 || free_count > 0;
  }
  if (ClTypeArtAlloca) {
    auto stack_args        = arg_collector->collectStack(allocas);
    const auto stack_count = mem_instrument->instrumentStack(stack_args);
    NumInstrumentedAlloca += stack_count;
    mod |= stack_count > 0;
  } else {
    // FIXME this is required by some unit tests
    NumInstrumentedAlloca += allocas.size();
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
}

void TypeArtPass::printStats(llvm::raw_ostream& out) {
  const unsigned max_string{12U};
  const unsigned max_val{5U};
  std::string line(22, '-');
  line += "\n";
  const auto make_format = [&](const char* desc, const auto val) {
    return format("%-*s: %*u\n", max_string, desc, max_val, val);
  };

  out << line;
  out << "   TypeArtPass\n";
  out << line;
  out << "Heap Memory\n";
  out << line;
  out << make_format("Malloc", NumInstrumentedMallocs.getValue());
  out << make_format("Free", NumInstrumentedFrees.getValue());
  out << line;
  out << "Stack Memory\n";
  out << line;
  out << make_format("Alloca", NumInstrumentedAlloca.getValue());
  out << line;
  out << "Global Memory\n";
  out << line;
  out << make_format("Global", NumInstrumentedGlobal.getValue());
  out << line;
  out.flush();
}

}  // namespace pass
}  // namespace typeart

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

static void registerClangPass(const llvm::PassManagerBuilder&, llvm::legacy::PassManagerBase& PM) {
  PM.add(new typeart::pass::TypeArtPass());
}
static RegisterStandardPasses RegisterClangPass(PassManagerBuilder::EP_OptimizerLast, registerClangPass);

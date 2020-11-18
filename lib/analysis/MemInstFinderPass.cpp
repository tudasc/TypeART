/*
 * MemInstFinderPass.cpp
 *
 *  Created on: Jun 3, 2018
 *      Author: ahueck
 */

#include "MemInstFinderPass.h"

#include "MemOpVisitor.h"
#include "filter/CGFilter.h"
#include "filter/CGForwardFilter.h"
#include "filter/CGInterface.h"
#include "filter/Filter.h"
#include "filter/StandardFilter.h"
#include "filter/StdForwardFilter.h"
#include "support/Logger.h"
#include "support/Table.h"
#include "support/TypeUtil.h"
#include "support/Util.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"

using namespace llvm;

#define DEBUG_TYPE "meminstanalysis"

namespace {
static RegisterPass<typeart::MemInstFinderPass> X("mem-inst-finder",                                // pass option
                                                  "Find heap and stack allocations in a program.",  // pass description
                                                  true,  // does not modify the CFG
                                                  true   // and it's an analysis
);
}  // namespace

static cl::opt<bool> ClFilterNonArrayAlloca("alloca-array-only", cl::desc("Only use alloca instructions of arrays."),
                                            cl::Hidden, cl::init(true));

static cl::opt<bool> ClFilterMallocAllocPair("malloc-store-filter",
                                             cl::desc("Filter allocs that get a store from a heap alloc."), cl::Hidden,
                                             cl::init(false));

static cl::opt<bool> ClCallFilter("call-filter",
                                  cl::desc("Filter alloca instructions that are passed to specific calls."), cl::Hidden,
                                  cl::init(false));

static cl::opt<bool> ClCallFilterDeep("call-filter-deep",
                                      cl::desc("If the CallFilter matches, we look if the value is passed as a void*."),
                                      cl::Hidden, cl::init(false));

static cl::opt<const char*> ClCallFilterGlob("call-filter-str", cl::desc("Filter alloca instructions based on string."),
                                             cl::Hidden, cl::init("*MPI_*"));

static cl::opt<bool> ClFilterGlobal("filter-globals", cl::desc("Filter globals of a module."), cl::Hidden,
                                    cl::init(true));

static cl::opt<std::string> ClFilterImpl("filter-impl", cl::desc("Select the filter implementation."), cl::Hidden,
                                         cl::init("default"));

static cl::opt<std::string> ClCGFile("cg-file", cl::desc("Location of CG to use."), cl::Hidden, cl::init(""));

STATISTIC(NumDetectedHeap, "Number of detected heap allocs");
STATISTIC(NumFilteredDetectedHeap, "Number of filtered heap allocs");
STATISTIC(NumDetectedAllocs, "Number of detected allocs");
STATISTIC(NumCallFilteredAllocs, "Number of call filtered allocs");
STATISTIC(NumFilteredMallocAllocs, "Number of  filtered  malloc-related allocs");
STATISTIC(NumFilteredNonArrayAllocs, "Number of call filtered allocs");
STATISTIC(NumDetectedGlobals, "Number of detected globals");
STATISTIC(NumFilteredGlobals, "Number of filtered globals");
STATISTIC(NumCallFilteredGlobals, "Number of filtered globals");

namespace typeart {

using namespace finder;

namespace filter {

static std::unique_ptr<Filter> make_filter(std::string id, std::string glob) {
  const bool deep = ClCallFilterDeep.getValue();
  if (id == "empty" || !ClCallFilter.getValue()) {
    LOG_DEBUG("Return no-op filter")
    return std::make_unique<NoOpFilter>();
  } else if (id == "CG" && !ClCGFile.empty()) {
    LOG_DEBUG("Return CG filter with CG @ " << ClCGFile.getValue())
    return std::make_unique<deprecated::CGFilter>(glob, deep, ClCGFile.getValue());
  } else if (id == "experimental::default") {
    LOG_DEBUG("Return experimental default filter")
    auto matcher = std::make_unique<filter::DefaultStringMatcher>(util::glob2regex(glob));
    // auto deep_matcher = std::make_unique<filter::DefaultStringMatcher>(util::glob2regex(glob));
    return std::make_unique<StandardForwardFilter>(std::move(matcher));
  } else if (id == "experimental::cg") {
    LOG_DEBUG("Return experimental CG filter with CG @ " << ClCGFile.getValue())
    return std::make_unique<CGForwardFilter>(glob, ClCGFile.getValue());
  } else {
    // default
    LOG_DEBUG("Return default filter")
    return std::make_unique<deprecated::StandardFilter>(glob, deep);
  }
}

CallFilter::CallFilter(const std::string& glob) : fImpl{make_filter(ClFilterImpl.getValue(), glob)} {
}

bool CallFilter::operator()(AllocaInst* in) {
  LOG_DEBUG("Analyzing value: " << util::dump(*in));
  fImpl->setMode(/*search mallocs = */ false);
  fImpl->setStartingFunction(in->getParent()->getParent());
  const auto filter_ = fImpl->filter(in);
  if (filter_) {
    LOG_DEBUG("Filtering value: " << util::dump(*in) << "\n");
  } else {
    LOG_DEBUG("Keeping value: " << util::dump(*in) << "\n");
  }
  return filter_;
}

bool CallFilter::operator()(GlobalValue* g) {
  LOG_DEBUG("Analyzing value: " << util::dump(*g));
  fImpl->setMode(/*search mallocs = */ false);
  fImpl->setStartingFunction(nullptr);
  const auto filter_ = fImpl->filter(g);
  if (filter_) {
    LOG_DEBUG("Filtering value: " << util::dump(*g) << "\n");
  } else {
    LOG_DEBUG("Keeping value: " << util::dump(*g) << "\n");
  }
  return filter_;
}

CallFilter& CallFilter::operator=(CallFilter&&) noexcept = default;

CallFilter::~CallFilter() = default;

}  // namespace filter

char MemInstFinderPass::ID = 0;

MemInstFinderPass::MemInstFinderPass() : llvm::ModulePass(ID), mOpsCollector(), filter(ClCallFilterGlob.getValue()) {
}

void MemInstFinderPass::getAnalysisUsage(llvm::AnalysisUsage& info) const {
  info.setPreservesAll();
}

bool MemInstFinderPass::runOnModule(Module& m) {
  mOpsCollector.visitModuleGlobals(m);
  auto& globals = mOpsCollector.globals;
  NumDetectedGlobals += globals.size();
  if (ClFilterGlobal && !ClFilterNonArrayAlloca) {
    globals.erase(
        llvm::remove_if(
            globals,
            [&](const auto gdata) {
              auto global     = gdata.global;
              const auto name = global->getName();
              if (name.startswith("llvm.") || name.startswith("__llvm_gcov") || name.startswith("__llvm_gcda")) {
                // 2nd and 3rd check: Check if the global is private gcov data.
                return true;
              }

              if (global->hasInitializer()) {
                auto* ini          = global->getInitializer();
                StringRef ini_name = util::dump(*ini);

                if (ini_name.contains("std::ios_base::Init")) {
                  return true;
                }
              }
              //              if (!g->hasInitializer()) {
              //                return true;
              //              }

              if (global->hasSection()) {
                StringRef Section = global->getSection();

                // Globals from llvm.metadata aren't emitted, do not instrument them.
                if (Section == "llvm.metadata") {
                  return true;
                }
                // Do not instrument globals from special LLVM sections.
                if (Section.find("__llvm") != StringRef::npos || Section.find("__LLVM") != StringRef::npos) {
                  return true;
                }
                // Check if the global is in the PGO counters section.
                //                auto OF = Triple(m.getTargetTriple()).getObjectFormat();
                //                if (Section.endswith(getInstrProfSectionName(IPSK_cnts, OF,
                //                /*AddSegmentInfo=*/false))) {
                //                  return true;
                //                }
              }

              if (global->getLinkage() == GlobalValue::ExternalLinkage ||
                  global->getLinkage() == GlobalValue::PrivateLinkage) {
                return true;
              }

              Type* t = global->getValueType();
              if (!t->isSized()) {
                return true;
              }

              if (t->isArrayTy()) {
                t = t->getArrayElementType();
              }
              if (auto structType = dyn_cast<StructType>(t)) {
                if (structType->isOpaque()) {
                  LOG_DEBUG("Encountered opaque struct " << t->getStructName() << " - skipping...");
                  return true;
                }
              }
              return false;
            }),
        globals.end());

    const auto beforeCallFilter = globals.size();
    NumFilteredGlobals          = NumDetectedGlobals - beforeCallFilter;

    globals.erase(llvm::remove_if(globals, [&](const auto g) { return filter(g.global); }), globals.end());

    NumCallFilteredGlobals = beforeCallFilter - globals.size();
    NumFilteredGlobals += NumCallFilteredGlobals;
  }

  return llvm::count_if(m.functions(), [&](auto& f) { return runOnFunc(f); }) > 0;
}  // namespace typeart

bool MemInstFinderPass::runOnFunc(llvm::Function& f) {
  if (f.isDeclaration() || f.getName().startswith("__typeart")) {
    return false;
  }

  LOG_DEBUG("Running on function: " << f.getName())

  mOpsCollector.visit(f);

  const auto checkAmbigiousMalloc = [&f](const MallocData& mallocData) {
    using namespace typeart::util::type;
    auto primaryBitcast = mallocData.primary;
    if (primaryBitcast) {
      const auto& bitcasts = mallocData.bitcasts;
      std::for_each(bitcasts.begin(), bitcasts.end(), [&](auto bitcastInst) {
        auto dest = bitcastInst->getDestTy();
        if (bitcastInst != primaryBitcast &&
            (!isVoidPtr(dest) && !isi64Ptr(dest) &&
             primaryBitcast->getDestTy() != dest)) {  // void* and i64* are used by LLVM
          // Second non-void* bitcast detected - semantics unclear
          LOG_WARNING("Encountered ambiguous pointer type in function: " << util::try_demangle(f));
          LOG_WARNING("  Allocation" << util::dump(*(mallocData.call)));
          LOG_WARNING("  Primary cast: " << util::dump(*primaryBitcast));
          LOG_WARNING("  Secondary cast: " << util::dump(*bitcastInst));
        }
      });
    }
  };

  NumDetectedAllocs += mOpsCollector.allocas.size();

  if (ClFilterNonArrayAlloca) {
    auto& allocs = mOpsCollector.allocas;
    allocs.erase(llvm::remove_if(allocs,
                                 [&](const auto& data) {
                                   if (!data.alloca->getAllocatedType()->isArrayTy() && data.array_size == 1) {
                                     ++NumFilteredNonArrayAllocs;
                                     return true;
                                   }
                                   return false;
                                 }),
                 allocs.end());
  }

  if (ClFilterMallocAllocPair) {
    auto& allocs  = mOpsCollector.allocas;
    auto& mallocs = mOpsCollector.mallocs;

    const auto filterMallocAllocPairing = [&mallocs](const auto alloc) {
      // Only look for the direct users of the alloc:
      // TODO is a deeper analysis required?
      for (auto inst : alloc->users()) {
        if (StoreInst* store = dyn_cast<StoreInst>(inst)) {
          const auto source = store->getValueOperand();
          if (isa<BitCastInst>(source)) {
            for (auto& mdata : mallocs) {
              // is it a bitcast we already collected? if yes, we can filter the alloc
              return std::any_of(mdata.bitcasts.begin(), mdata.bitcasts.end(),
                                 [&source](const auto bcast) { return bcast == source; });
            }
          } else if (isa<CallInst>(source)) {
            return std::any_of(mallocs.begin(), mallocs.end(),
                               [&source](const auto& mdata) { return mdata.call == source; });
          }
        }
      }
      return false;
    };

    allocs.erase(llvm::remove_if(allocs,
                                 [&](const auto& data) {
                                   if (filterMallocAllocPairing(data.alloca)) {
                                     ++NumFilteredMallocAllocs;
                                     return true;
                                   }
                                   return false;
                                 }),
                 allocs.end());
  }

  if (ClCallFilter) {
    auto& allocs = mOpsCollector.allocas;
    allocs.erase(llvm::remove_if(allocs,
                                 [&](const auto& data) {
                                   if (filter(data.alloca)) {
                                     ++NumCallFilteredAllocs;
                                     return true;
                                   }
                                   return false;
                                 }),
                 allocs.end());
    //    LOG_DEBUG(allocs.size() << " allocas to instrument : " << util::dump(allocs));
  }

  auto& mallocs = mOpsCollector.mallocs;
  NumDetectedHeap += mallocs.size();

  for (const auto& mallocData : mallocs) {
    checkAmbigiousMalloc(mallocData);
  }

  FunctionData d{mOpsCollector.mallocs, mOpsCollector.frees, mOpsCollector.allocas};
  functionMap[&f] = d;

  mOpsCollector.clear();

  return false;
}  // namespace typeart

bool MemInstFinderPass::doFinalization(llvm::Module&) {
  if (AreStatisticsEnabled()) {
    auto& out = llvm::errs();
    printStats(out);
  }
  return false;
}

void MemInstFinderPass::printStats(llvm::raw_ostream& out) {
  auto all_stack          = double(NumDetectedAllocs.getValue());
  auto nonarray_stack     = double(NumFilteredNonArrayAllocs.getValue());
  auto malloc_alloc_stack = double(NumFilteredMallocAllocs.getValue());
  auto call_filter_stack  = double(NumCallFilteredAllocs.getValue());

  const auto call_filter_stack_p =
      (call_filter_stack / std::max<double>(1.0, all_stack - nonarray_stack - malloc_alloc_stack)) * 100.0;

  const auto call_filter_heap_p =
      (double(NumFilteredDetectedHeap.getValue()) / std::max<double>(1.0, double(NumDetectedHeap.getValue()))) * 100.0;

  const auto call_filter_global_p =
      (double(NumCallFilteredGlobals.getValue()) / std::max(1.0, double(NumDetectedGlobals.getValue()))) * 100.0;

  const auto call_filter_global_nocallfilter_p =
      (double(NumFilteredGlobals.getValue()) / std::max(1.0, double(NumDetectedGlobals.getValue()))) * 100.0;

  Table stats("MemInstFinderPass");
  stats.wrap_header = true;
  stats.wrap_length = true;
  stats.put(Row::make("Filter string", ClCallFilterGlob.getValue()));
  stats.put(Row::make_row("> Heap Memory"));
  stats.put(Row::make("Heap alloc", NumDetectedHeap.getValue()));
  stats.put(Row::make("Heap call filtered %", call_filter_heap_p));
  stats.put(Row::make_row("> Stack Memory"));
  stats.put(Row::make("Alloca", all_stack));
  stats.put(Row::make("Stack call filtered %", call_filter_stack_p));
  stats.put(Row::make_row("> Global Memory"));
  stats.put(Row::make("Global", NumDetectedGlobals.getValue()));
  stats.put(Row::make("Global filter total", NumFilteredGlobals.getValue()));
  stats.put(Row::make("Global call filtered %", call_filter_global_p));
  stats.put(Row::make("Global filtered %", call_filter_global_nocallfilter_p));

  stats.print(out);
}

bool MemInstFinderPass::hasFunctionData(Function* f) const {
  auto iter = functionMap.find(f);
  return iter != functionMap.end();
}

const FunctionData& MemInstFinderPass::getFunctionData(Function* f) const {
  auto iter = functionMap.find(f);
  return iter->second;
}

const llvm::SmallVectorImpl<GlobalData>& MemInstFinderPass::getModuleGlobals() const {
  return mOpsCollector.globals;
}

}  // namespace typeart

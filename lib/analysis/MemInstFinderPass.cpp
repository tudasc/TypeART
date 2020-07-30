/*
 * MemInstFinderPass.cpp
 *
 *  Created on: Jun 3, 2018
 *      Author: ahueck
 */

#include "MemInstFinderPass.h"

#include "CGFilter.h"
#include "Filter.h"
#include "MemOpVisitor.h"
#include "StandardFilter.h"
#include "support/Logger.h"
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
                                             cl::Hidden, cl::init("MPI_*"));

static cl::opt<bool> ClFilterGlobal("filter-globals", cl::desc("Filter globals of a module."), cl::Hidden,
                                    cl::init(true));

static cl::opt<std::string> ClCGFile("cg-file", cl::desc("Location of CG to use."), cl::Hidden, cl::init(""));

static cl::opt<std::string> ClFilterImpl("filter-impl", cl::desc("Select the filter implementation."), cl::Hidden,
                                         cl::init("default"));

static cl::opt<std::string> ClFilterFile("typeart-filter-outfile", cl::desc("Location of the generated alloc file."),
                                         cl::Hidden, cl::init("allocs-filtered.yaml"));

STATISTIC(NumDetectedHeap, "Number of detected heap allocs");
STATISTIC(NumFilteredDetectedHeap, "Number of filtered heap allocs");
STATISTIC(NumDetectedFree, "Number of detected frees");
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

static std::unique_ptr<FilterBase> make_filter(std::string id, std::string glob) {
  const bool deep = ClCallFilterDeep.getValue();
  if (id == "cg") {
    LOG_FATAL("Demand cg filter")
    return std::make_unique<CGFilterImpl>(glob, deep, ClCGFile.getValue());
  } else if (id == "empty" || !ClCallFilter.getValue()) {
    LOG_FATAL("Demand empty filter")
    return std::make_unique<FilterBase>(glob, deep);
  } else {
    // default
    LOG_FATAL("Default filter")
    return std::make_unique<FilterImpl>(glob, deep);
  }
}

CallFilter::CallFilter(const std::string& glob, ModuleDataManager& m)
    : fImpl{make_filter(ClFilterImpl.getValue(), glob)}, m(m) {
}

bool CallFilter::operator()(const AllocaData& adata) {
  auto* in = adata.alloca;
  LOG_DEBUG("Analyzing value: " << util::dump(*in));
  fImpl->setMode(/*search mallocs = */ false);
  fImpl->setStartingFunction(in->getParent()->getParent());
  const auto filter_ = fImpl->filter(in);
  if (filter_) {
    LOG_DEBUG("Filtering value: " << util::dump(*in) << "\n");
    auto i   = adata.alloca;
    auto dbg = util::getDebugVar(*i);
    if (dbg != nullptr) {
      if (dbg->getName() == "gnewdt") {
        m.putStack(adata, -1, "Keep");
        fImpl->clear_trace();
        return false;
      }
    }
    m.putStack(adata, -1, "CallFilter " + fImpl->reason());
  } else {
    LOG_DEBUG("Keeping value: " << util::dump(*in) << "\n");
    m.putStack(adata, -1, "Keep " + fImpl->reason());
  }
  fImpl->clear_trace();
  return filter_;
}

bool CallFilter::operator()(GlobalVariable* g) {
  LOG_DEBUG("Analyzing value: " << util::dump(*g));
  fImpl->setMode(/*search mallocs = */ false);
  fImpl->setStartingFunction(nullptr);
  const auto filter_ = fImpl->filter(g);
  if (filter_) {
    LOG_DEBUG("Filtering value: " << util::dump(*g) << "\n");
    m.putGlobal(g, -1, "CallFilter " + fImpl->reason());
  } else {
    LOG_DEBUG("Keeping value: " << util::dump(*g) << "\n");
  }
  fImpl->clear_trace();
  return filter_;
}

}  // namespace filter

char MemInstFinderPass::ID = 0;

MemInstFinderPass::MemInstFinderPass()
    : llvm::ModulePass(ID),
      mOpsCollector(),
      data_m(ClFilterFile.getValue()),
      filter(ClCallFilterGlob.getValue(), data_m) {
}

void MemInstFinderPass::getAnalysisUsage(llvm::AnalysisUsage& info) const {
  info.setPreservesAll();
}

bool MemInstFinderPass::runOnModule(Module& m) {
  data_m.load();
  data_m.lookupModule(m);

  mOpsCollector.visitModuleGlobals(m);
  auto& globals = mOpsCollector.listGlobals;
  NumDetectedGlobals += globals.size();
  if (ClFilterGlobal && !ClFilterNonArrayAlloca) {
    globals.erase(
        llvm::remove_if(
            globals,
            [&](const auto g) {
              const auto name = g->getName();
              if (name.startswith("llvm.") || name.startswith("__llvm_gcov") || name.startswith("__llvm_gcda")) {
                // 2nd and 3rd check: Check if the global is private gcov data.
                return true;
              }

              if (g->hasInitializer()) {
                auto* ini          = g->getInitializer();
                StringRef ini_name = util::dump(*ini);

                if (ini_name.contains("std::ios_base::Init")) {
                  return true;
                }
              }
              //              if (!g->hasInitializer()) {
              //                return true;
              //              }

              if (g->hasSection()) {
                StringRef Section = g->getSection();

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

              if (g->getLinkage() == GlobalValue::ExternalLinkage || g->getLinkage() == GlobalValue::PrivateLinkage) {
                return true;
              }

              Type* t = g->getValueType();
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

    globals.erase(llvm::remove_if(globals, [&](const auto g) { return filter(g); }), globals.end());

    NumCallFilteredGlobals = beforeCallFilter - globals.size();
    NumFilteredGlobals += NumCallFilteredGlobals;
  }

  return llvm::count_if(m.functions(), [&](auto& f) { return runOnFunc(f); }) > 0;
}  // namespace typeart

bool MemInstFinderPass::runOnFunc(llvm::Function& f) {
  if (f.isDeclaration() || f.getName().startswith("__typeart")) {
    return false;
  }

  const auto FID = data_m.lookupFunction(f);

  mOpsCollector.visit(f);

  LOG_DEBUG("Running on function: " << f.getName())

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

  NumDetectedAllocs += mOpsCollector.listAlloca.size();

  if (ClFilterNonArrayAlloca) {
    auto& allocs = mOpsCollector.listAlloca;
    allocs.erase(llvm::remove_if(allocs,
                                 [&](const auto& data) {
                                   if (!data.alloca->getAllocatedType()->isArrayTy() && data.arraySize == 1) {
                                     ++NumFilteredNonArrayAllocs;
                                     return true;
                                   }
                                   return false;
                                 }),
                 allocs.end());
  }

  if (ClFilterMallocAllocPair) {
    auto& allocs = mOpsCollector.listAlloca;
    auto& mlist  = mOpsCollector.listMalloc;

    const auto filterMallocAllocPairing = [&mlist](const auto alloc) {
      // Only look for the direct users of the alloc:
      // TODO is a deeper analysis required?
      for (auto inst : alloc->users()) {
        if (StoreInst* store = dyn_cast<StoreInst>(inst)) {
          const auto source = store->getValueOperand();
          if (isa<BitCastInst>(source)) {
            for (auto& mdata : mlist) {
              // is it a bitcast we already collected? if yes, we can filter the alloc
              return std::any_of(mdata.bitcasts.begin(), mdata.bitcasts.end(),
                                 [&source](const auto bcast) { return bcast == source; });
            }
          } else if (isa<CallInst>(source)) {
            return std::any_of(mlist.begin(), mlist.end(),
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
    auto& allocs = mOpsCollector.listAlloca;
    allocs.erase(llvm::remove_if(allocs,
                                 [&](const auto& data) {
                                   if (filter(data)) {
                                     ++NumCallFilteredAllocs;
                                     return true;
                                   }
                                   return false;
                                 }),
                 allocs.end());
    //    LOG_DEBUG(allocs.size() << " allocas to instrument : " << util::dump(allocs));
  }

  auto& mallocs = mOpsCollector.listMalloc;
  NumDetectedHeap += mallocs.size();

  for (const auto& mallocData : mallocs) {
    checkAmbigiousMalloc(mallocData);
  }

  NumDetectedFree += mOpsCollector.listFree.size();

  FunctionData d{mOpsCollector.listMalloc, mOpsCollector.listFree, mOpsCollector.listAlloca};
  functionMap[&f] = d;

  mOpsCollector.clear();

  return false;
}  // namespace typeart

bool MemInstFinderPass::doFinalization(llvm::Module&) {
  data_m.clearEmpty();
  data_m.store();
  if (AreStatisticsEnabled()) {
    auto& out = llvm::errs();
    printStats(out);
  }
  return false;
}

void MemInstFinderPass::printStats(llvm::raw_ostream& out) {
  const unsigned max_string{28u};
  const unsigned max_val{5u};
  std::string line(42, '-');
  line += "\n";
  const auto make_format = [&](const char* desc, const auto val) {
    return format("%-*s: %*.1f\n", max_string, desc, max_val, val);
  };

  auto all_stack          = double(NumDetectedAllocs.getValue());
  auto nonarray_stack     = double(NumFilteredNonArrayAllocs.getValue());
  auto malloc_alloc_stack = double(NumFilteredMallocAllocs.getValue());
  auto call_filter_stack  = double(NumCallFilteredAllocs.getValue());

  out << line;
  out << "   MemInstFinderPass\n";
  out << line;
  out << "Heap Memory\n";
  out << line;
  out << make_format("Heap alloc", double(NumDetectedHeap.getValue()));
  out << make_format(
      "% call filtered",
      (double(NumFilteredDetectedHeap.getValue()) / std::max(1.0, double(NumDetectedHeap.getValue()))) * 100.0);
  out << line;
  out << "Free Memory\n";
  out << line;
  out << make_format("Frees", double(NumDetectedFree.getValue()));
  out << line;
  out << "Stack Memory\n";
  out << line;
  out << make_format("Alloca", all_stack);
  out << make_format("% non array filtered", (nonarray_stack / std::max(1.0, all_stack)) * 100.0);
  out << make_format("% malloc-alloc filtered",
                     (malloc_alloc_stack / std::max(1.0, all_stack - nonarray_stack)) * 100.0);
  out << make_format("% call filtered",
                     (call_filter_stack / std::max(1.0, all_stack - nonarray_stack - malloc_alloc_stack)) * 100.0);
  out << line;
  out << "Global Memory\n";
  out << line;
  out << make_format("Global", double(NumDetectedGlobals.getValue()));
  out << make_format("Global total filtered", double(NumFilteredGlobals.getValue()));
  out << make_format("Global Call Filtered", double(NumCallFilteredGlobals.getValue()));
  out << make_format(
      "% global call filtered",
      (double(NumCallFilteredGlobals.getValue()) / std::max(1.0, double(NumDetectedGlobals.getValue()))) * 100.0);
  out << make_format(
      "% global filtered",
      (double(NumFilteredGlobals.getValue()) / std::max(1.0, double(NumDetectedGlobals.getValue()))) * 100.0);
  out << line;
  out.flush();
}

bool MemInstFinderPass::hasFunctionData(Function* f) const {
  auto iter = functionMap.find(f);
  return iter != functionMap.end();
}

const FunctionData& MemInstFinderPass::getFunctionData(Function* f) const {
  auto iter = functionMap.find(f);
  return iter->second;
}

const llvm::SmallVector<llvm::GlobalVariable*, 8>& MemInstFinderPass::getModuleGlobals() const {
  return mOpsCollector.listGlobals;
}

}  // namespace typeart

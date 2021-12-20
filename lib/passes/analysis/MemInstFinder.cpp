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

#include "MemInstFinder.h"

#include "MemOpVisitor.h"
#include "analysis/MemOpData.h"
#include "filter/CGForwardFilter.h"
#include "filter/CGInterface.h"
#include "filter/Filter.h"
#include "filter/Matcher.h"
#include "filter/StandardFilter.h"
#include "filter/StdForwardFilter.h"
#include "support/Logger.h"
#include "support/Table.h"
#include "support/TypeUtil.h"
#include "support/Util.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/PassAnalysisSupport.h"
#include "llvm/PassSupport.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "MemInstFinder"
STATISTIC(NumDetectedHeap, "Number of detected heap allocs");
STATISTIC(NumFilteredDetectedHeap, "Number of filtered heap allocs");
STATISTIC(NumDetectedAllocs, "Number of detected allocs");
STATISTIC(NumCallFilteredAllocs, "Number of call filtered allocs");
STATISTIC(NumFilteredMallocAllocs, "Number of  filtered  malloc-related allocs");
STATISTIC(NumFilteredNonArrayAllocs, "Number of call filtered allocs");
STATISTIC(NumDetectedGlobals, "Number of detected globals");
STATISTIC(NumFilteredGlobals, "Number of filtered globals");
STATISTIC(NumCallFilteredGlobals, "Number of filtered globals");

namespace typeart::analysis {

namespace filter {
class CallFilter {
  std::unique_ptr<typeart::filter::Filter> fImpl;

 public:
  explicit CallFilter(const MemInstFinderConfig& config);
  CallFilter(const CallFilter&) = delete;
  CallFilter(CallFilter&&)      = default;
  bool operator()(llvm::AllocaInst*);
  bool operator()(llvm::GlobalValue*);
  CallFilter& operator=(CallFilter&&) noexcept;
  CallFilter& operator=(const CallFilter&) = delete;
  virtual ~CallFilter();
};

}  // namespace filter

namespace filter {

namespace detail {
static std::unique_ptr<typeart::filter::Filter> make_filter(const MemInstFinderConfig& config) {
  using namespace typeart::filter;
  const bool deep        = config.filter.ClCallFilterDeep;
  const std::string id   = config.filter.ClCallFilterImpl;
  const std::string glob = config.filter.ClCallFilterGlob;

  if (id == "empty" || !config.filter.ClUseCallFilter) {
    LOG_DEBUG("Return no-op filter")
    return std::make_unique<NoOpFilter>();
  } else if (id == "deprecated::default") {
    // default
    LOG_DEBUG("Return deprecated default filter")
    return std::make_unique<deprecated::StandardFilter>(glob, deep);
  } else if (id == "cg" || id == "experimental::cg") {
    if (config.filter.ClCallFilterCGFile.empty()) {
      LOG_FATAL("CG File not set!");
      std::exit(1);
    }
    LOG_DEBUG("Return CG filter with CG file @ " << config.filter.ClCallFilterCGFile)
    auto json_cg = JSONCG::getJSON(config.filter.ClCallFilterCGFile);
    auto matcher = std::make_unique<DefaultStringMatcher>(util::glob2regex(glob));
    return std::make_unique<CGForwardFilter>(glob, std::move(json_cg), std::move(matcher));
  } else {
    LOG_DEBUG("Return default filter")
    auto matcher         = std::make_unique<DefaultStringMatcher>(util::glob2regex(glob));
    const auto deep_glob = config.filter.ClCallFilterDeepGlob;
    auto deep_matcher    = std::make_unique<DefaultStringMatcher>(util::glob2regex(deep_glob));
    return std::make_unique<StandardForwardFilter>(std::move(matcher), std::move(deep_matcher));
  }
}
}  // namespace detail

CallFilter::CallFilter(const MemInstFinderConfig& config) : fImpl{detail::make_filter(config)} {
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

class MemInstFinderPass : public MemInstFinder {
 private:
  MemOpVisitor mOpsCollector;
  filter::CallFilter filter;
  llvm::DenseMap<const llvm::Function*, FunctionData> functionMap;
  MemInstFinderConfig config;

 public:
  explicit MemInstFinderPass(const MemInstFinderConfig&);
  bool runOnModule(llvm::Module&) override;
  bool hasFunctionData(const llvm::Function&) const override;
  const FunctionData& getFunctionData(const llvm::Function&) const override;
  const GlobalDataList& getModuleGlobals() const override;
  void printStats(llvm::raw_ostream&) const override;
  // void configure(MemInstFinderConfig&) override;
  ~MemInstFinderPass() = default;

 private:
  bool runOnFunction(llvm::Function&);
};

MemInstFinderPass::MemInstFinderPass(const MemInstFinderConfig& config)
    : mOpsCollector(config.ClTypeArtAlloca, !config.ClIgnoreHeap), filter(config), config(config) {
}

bool MemInstFinderPass::runOnModule(Module& module) {
  mOpsCollector.visitModuleGlobals(module);
  auto& globals = mOpsCollector.globals;
  NumDetectedGlobals += globals.size();
  if (config.filter.ClFilterGlobal) {
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

    globals.erase(llvm::remove_if(globals, [&](const auto global) { return filter(global.global); }), globals.end());

    NumCallFilteredGlobals = beforeCallFilter - globals.size();
    NumFilteredGlobals += NumCallFilteredGlobals;
  }

  return llvm::count_if(module.functions(), [&](auto& function) { return runOnFunction(function); }) > 0;
}  // namespace typeart

bool MemInstFinderPass::runOnFunction(llvm::Function& function) {
  if (function.isDeclaration() || function.getName().startswith("__typeart")) {
    return false;
  }

  LOG_DEBUG("Running on function: " << function.getName())

  mOpsCollector.visit(function);

  const auto checkAmbigiousMalloc = [&function](const MallocData& mallocData) {
    using namespace typeart::util::type;
    auto primaryBitcast = mallocData.primary;
    if (primaryBitcast != nullptr) {
      const auto& bitcasts = mallocData.bitcasts;
      std::for_each(bitcasts.begin(), bitcasts.end(), [&](auto bitcastInst) {
        auto dest = bitcastInst->getDestTy();
        if (bitcastInst != primaryBitcast &&
            (!isVoidPtr(dest) && !isi64Ptr(dest) &&
             primaryBitcast->getDestTy() != dest)) {  // void* and i64* are used by LLVM
          // Second non-void* bitcast detected - semantics unclear
          LOG_WARNING("Encountered ambiguous pointer type in function: " << util::try_demangle(function));
          LOG_WARNING("  Allocation" << util::dump(*(mallocData.call)));
          LOG_WARNING("  Primary cast: " << util::dump(*primaryBitcast));
          LOG_WARNING("  Secondary cast: " << util::dump(*bitcastInst));
        }
      });
    }
  };

  NumDetectedAllocs += mOpsCollector.allocas.size();

  LOG_DEBUG(config.filter.ClFilterNonArrayAlloca);
  if (config.filter.ClFilterNonArrayAlloca) {
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

  if (config.filter.ClFilterMallocAllocPair) {
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

  if (config.filter.ClUseCallFilter) {
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

  FunctionData data{mOpsCollector.mallocs, mOpsCollector.frees, mOpsCollector.allocas};
  functionMap[&function] = data;

  mOpsCollector.clear();

  return true;
}  // namespace typeart

void MemInstFinderPass::printStats(llvm::raw_ostream& out) const {
  auto all_stack          = double(NumDetectedAllocs);
  auto nonarray_stack     = double(NumFilteredNonArrayAllocs);
  auto malloc_alloc_stack = double(NumFilteredMallocAllocs);
  auto call_filter_stack  = double(NumCallFilteredAllocs);

  const auto call_filter_stack_p =
      (call_filter_stack / std::max<double>(1.0, all_stack - nonarray_stack - malloc_alloc_stack)) * 100.0;

  const auto call_filter_heap_p =
      (double(NumFilteredDetectedHeap) / std::max<double>(1.0, double(NumDetectedHeap))) * 100.0;

  const auto call_filter_global_p =
      (double(NumCallFilteredGlobals) / std::max(1.0, double(NumDetectedGlobals))) * 100.0;

  const auto call_filter_global_nocallfilter_p =
      (double(NumFilteredGlobals) / std::max(1.0, double(NumDetectedGlobals))) * 100.0;

  Table stats("MemInstFinderPass");
  stats.wrap_header = true;
  stats.wrap_length = true;
  stats.put(Row::make("Filter string", config.filter.ClCallFilterGlob));
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

  std::ostringstream stream;
  stats.print(stream);
  out << stream.str();
}

bool MemInstFinderPass::hasFunctionData(const Function& function) const {
  auto iter = functionMap.find(&function);
  return iter != functionMap.end();
}

const FunctionData& MemInstFinderPass::getFunctionData(const Function& function) const {
  auto iter = functionMap.find(&function);
  return iter->second;
}

const GlobalDataList& MemInstFinderPass::getModuleGlobals() const {
  return mOpsCollector.globals;
}

std::unique_ptr<MemInstFinder> create_finder(const MemInstFinderConfig& config) {
  return std::make_unique<MemInstFinderPass>(config);
}

}  // namespace typeart::analysis

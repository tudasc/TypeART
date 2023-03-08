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
#include "MemInstFinder.h"

#include "MemOpVisitor.h"
#include "analysis/MemOpData.h"
#include "filter/ACGForwardFilter.h"
#include "filter/CGForwardFilter.h"
#include "filter/CGInterface.h"
#include "filter/Filter.h"
#include "filter/Matcher.h"
#include "filter/StdForwardFilter.h"
#include "support/Configuration.h"
#include "support/JSONHelper.h"
#include "support/Logger.h"
#include "support/MetaCG.h"
#include "support/MetaCGExtension.h"
#include "support/Table.h"
#include "support/TypeUtil.h"
#include "support/Util.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "MemInstFinder"
ALWAYS_ENABLED_STATISTIC(NumDetectedHeap, "Number of detected heap allocs");
ALWAYS_ENABLED_STATISTIC(NumFilteredDetectedHeap, "Number of filtered heap allocs");
ALWAYS_ENABLED_STATISTIC(NumDetectedAllocs, "Number of detected allocs");
ALWAYS_ENABLED_STATISTIC(NumFilteredPointerAllocs, "Number of filtered pointer allocs");
ALWAYS_ENABLED_STATISTIC(NumCallFilteredAllocs, "Number of call filtered allocs");
ALWAYS_ENABLED_STATISTIC(NumFilteredMallocAllocs, "Number of  filtered  malloc-related allocs");
ALWAYS_ENABLED_STATISTIC(NumFilteredNonArrayAllocs, "Number of filtered non-array allocs");
ALWAYS_ENABLED_STATISTIC(NumDetectedGlobals, "Number of detected globals");
ALWAYS_ENABLED_STATISTIC(NumFilteredGlobals, "Number of filtered globals");
ALWAYS_ENABLED_STATISTIC(NumCallFilteredGlobals, "Number of filtered globals");

namespace typeart::analysis {

struct MemInstFinderConfig {
  struct Filter {
    bool useCallFilter{false};
    FilterImplementation implementation{FilterImplementation::standard};
    std::string callFilterGlob{"*MPI_*"};
    std::string callFilterDeepGlob{"MPI_*"};
    std::string callFilterCgFile{};
    bool filterNonArrayAlloca{false};
    bool filterMallocAllocPair{false};
    bool filterGlobal{true};
    bool filterPointerAlloca{false};
  };

  bool collect_heap{false};
  bool collect_alloca{false};
  bool collect_global{false};
  Filter filter;
};

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
  using namespace typeart::util::metacg;

  const auto filter_id   = config.filter.useCallFilter ? config.filter.implementation : FilterImplementation::none;
  const std::string glob = config.filter.callFilterGlob;

  switch (filter_id) {
    case FilterImplementation::none: {
      LOG_DEBUG("Return no-op filter")
      return std::make_unique<NoOpFilter>();
    }

    case FilterImplementation::standard: {
      LOG_DEBUG("Return default filter")
      auto matcher         = std::make_unique<DefaultStringMatcher>(util::glob2regex(glob));
      const auto deep_glob = config.filter.callFilterDeepGlob;
      auto deep_matcher    = std::make_unique<DefaultStringMatcher>(util::glob2regex(deep_glob));
      return std::make_unique<StandardForwardFilter>(std::move(matcher), std::move(deep_matcher));
    }

    case FilterImplementation::cg: {
      LOG_DEBUG("Return CG filter with CG file @ " << config.filter.callFilterCgFile)
      auto json_cg = std::make_unique<JSONCG>(util::getJSON(config.filter.callFilterCgFile).get());
      auto matcher = std::make_unique<DefaultStringMatcher>(util::glob2regex(glob));
      return std::make_unique<CGForwardFilter>(glob, std::move(json_cg), std::move(matcher));
    }

    case FilterImplementation::acg: {
      LOG_DEBUG("Return ACG filter with CG file @ " << config.filter.callFilterCgFile)
      auto json_cg = util::getJSON<JSONACG>(config.filter.callFilterCgFile).get();
      const Regex TargetMatcher{util::glob2regex(glob), Regex::NoFlags};
      return std::make_unique<ACGForwardFilter>(createDatabase(TargetMatcher, json_cg));
    }
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
  [[nodiscard]] bool hasFunctionData(const llvm::Function&) const override;
  [[nodiscard]] const FunctionData& getFunctionData(const llvm::Function&) const override;
  const GlobalDataList& getModuleGlobals() const override;
  void printStats(llvm::raw_ostream&) const override;
  // void configure(MemInstFinderConfig&) override;
  ~MemInstFinderPass() = default;

 private:
  bool runOnFunction(llvm::Function&);
};

MemInstFinderPass::MemInstFinderPass(const MemInstFinderConfig& config)
    : mOpsCollector(config.collect_alloca, config.collect_heap), filter(config), config(config) {
}

bool MemInstFinderPass::runOnModule(Module& module) {
  mOpsCollector.collectGlobals(module);
  auto& globals = mOpsCollector.globals;
  NumDetectedGlobals += globals.size();
  if (config.filter.filterGlobal) {
    globals.erase(llvm::remove_if(
                      globals,
                      [&](const auto gdata) {  // NOLINT
                        GlobalVariable* global = gdata.global;
                        const auto name        = global->getName();

                        LOG_DEBUG("Analyzing global: " << name);

                        if (name.empty()) {
                          return true;
                        }

                        if (name.startswith("llvm.") || name.startswith("__llvm_gcov") ||
                            name.startswith("__llvm_gcda") || name.startswith("__profn")) {
                          // 2nd and 3rd check: Check if the global is private gcov data (--coverage).
                          LOG_DEBUG("LLVM startswith \"llvm\"")
                          return true;
                        }

                        if (name.startswith("___asan") || name.startswith("__msan") || name.startswith("__tsan")) {
                          LOG_DEBUG("LLVM startswith \"sanitizer\"")
                          return true;
                        }

                        if (global->hasInitializer()) {
                          auto* ini            = global->getInitializer();
                          std::string ini_name = util::dump(*ini);

                          if (llvm::StringRef(ini_name).contains("std::ios_base::Init")) {
                            LOG_DEBUG("std::ios");
                            return true;
                          }
                        }

                        if (global->hasSection()) {
                          // for instance, filters:
                          //   a) (Coverage) -fprofile-instr-generate -fcoverage-mapping
                          //   b) (PGO) -fprofile-instr-generate
                          StringRef Section = global->getSection();
                          // Globals from llvm.metadata aren't emitted, do not instrument them.
                          if (Section == "llvm.metadata") {
                            LOG_DEBUG("metadata");
                            return true;
                          }
                          // Do not instrument globals from special LLVM sections.
                          if (Section.find("__llvm") != StringRef::npos || Section.find("__LLVM") != StringRef::npos) {
                            LOG_DEBUG("llvm section");
                            return true;
                          }
                        }

                        if ((global->getLinkage() == GlobalValue::ExternalLinkage && global->isDeclaration())) {
                          LOG_DEBUG("Linkage: External");
                          return true;
                        }

                        Type* global_type = global->getValueType();
                        if (!global_type->isSized()) {
                          LOG_DEBUG("not sized");
                          return true;
                        }

                        if (global_type->isArrayTy()) {
                          global_type = global_type->getArrayElementType();
                        }
                        if (auto structType = dyn_cast<StructType>(global_type)) {
                          if (structType->isOpaque()) {
                            LOG_DEBUG("Encountered opaque struct " << global_type->getStructName() << " - skipping...");
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

  mOpsCollector.collect(function);

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

  if (config.filter.filterNonArrayAlloca) {
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

  if (config.filter.filterMallocAllocPair) {
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

  if (config.filter.filterPointerAlloca) {
    auto& allocs = mOpsCollector.allocas;
    allocs.erase(llvm::remove_if(allocs,
                                 [&](const auto& data) {
                                   auto alloca = data.alloca;
                                   if (!data.is_vla && isa<llvm::PointerType>(alloca->getAllocatedType())) {
                                     ++NumFilteredPointerAllocs;
                                     return true;
                                   }
                                   return false;
                                 }),
                 allocs.end());
  }

  if (config.filter.useCallFilter) {
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
  auto all_stack            = double(NumDetectedAllocs);
  auto nonarray_stack       = double(NumFilteredNonArrayAllocs);
  auto malloc_alloc_stack   = double(NumFilteredMallocAllocs);
  auto call_filter_stack    = double(NumCallFilteredAllocs);
  auto filter_pointer_stack = double(NumFilteredPointerAllocs);

  const auto call_filter_stack_p =
      (call_filter_stack /
       std::max<double>(1.0, all_stack - nonarray_stack - malloc_alloc_stack - filter_pointer_stack)) *
      100.0;

  const auto call_filter_heap_p =
      (double(NumFilteredDetectedHeap) / std::max<double>(1.0, double(NumDetectedHeap))) * 100.0;

  const auto call_filter_global_p =
      (double(NumCallFilteredGlobals) / std::max(1.0, double(NumDetectedGlobals))) * 100.0;

  const auto call_filter_global_nocallfilter_p =
      (double(NumFilteredGlobals) / std::max(1.0, double(NumDetectedGlobals))) * 100.0;

  Table stats("MemInstFinderPass");
  stats.wrap_header = true;
  stats.wrap_length = true;
  stats.put(Row::make("Filter string", config.filter.callFilterGlob));
  stats.put(Row::make_row("> Heap Memory"));
  stats.put(Row::make("Heap alloc", NumDetectedHeap.getValue()));
  stats.put(Row::make("Heap call filtered %", call_filter_heap_p));
  stats.put(Row::make_row("> Stack Memory"));
  stats.put(Row::make("Alloca", all_stack));
  stats.put(Row::make("Stack call filtered %", call_filter_stack_p));
  stats.put(Row::make("Alloca of pointer discarded", filter_pointer_stack));
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

std::unique_ptr<MemInstFinder> create_finder(const config::Configuration& config) {
  using typeart::config::ConfigStdArgs;
  const auto meminst_config = [&config]() {
    return analysis::MemInstFinderConfig{
        config[ConfigStdArgs::heap],                                                                    //
        config[ConfigStdArgs::stack],                                                                   //
        config[ConfigStdArgs::global],                                                                  //
        analysis::MemInstFinderConfig::Filter{config[ConfigStdArgs::filter],                            //
                                              config[ConfigStdArgs::filter_impl],                       //
                                              config[ConfigStdArgs::filter_glob],                       //
                                              config[ConfigStdArgs::filter_glob_deep],                  //
                                              config[ConfigStdArgs::filter_cg_file],                    //
                                              config[ConfigStdArgs::analysis_filter_alloca_non_array],  //
                                              config[ConfigStdArgs::analysis_filter_heap_alloc],        //
                                              config[ConfigStdArgs::analysis_filter_global],            //
                                              config[ConfigStdArgs::analysis_filter_pointer_alloc]}};   //
  }();
  return std::make_unique<MemInstFinderPass>(meminst_config);
}

}  // namespace typeart::analysis

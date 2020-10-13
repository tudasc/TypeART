/*
 * MemInstFinderPass.cpp
 *
 *  Created on: Jun 3, 2018
 *      Author: ahueck
 */

#include "MemInstFinderPass.h"

#include "MemOpVisitor.h"
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

class CallFilter::FilterImpl {
  const std::string call_regex;
  bool malloc_mode{false};
  llvm::Function* start_f{nullptr};
  int depth{0};

 public:
  explicit FilterImpl(const std::string& glob) : call_regex(util::glob2regex(glob)) {
  }

  void setMode(bool search_malloc) {
    malloc_mode = search_malloc;
  }

  void setStartingFunction(llvm::Function* start) {
    start_f = start;
    depth   = 0;
  }

  bool filter(Value* in) {
    if (in == nullptr) {
      LOG_DEBUG("Called with nullptr");
      return false;
    }

    if (depth == 15) {
      return false;
    }

    const auto match = [&](auto callee) -> bool {
      const auto name = FilterImpl::getName(callee);
      return util::regex_matches(call_regex, name);
    };

    llvm::SmallPtrSet<Value*, 16> visited_set;
    llvm::SmallVector<Value*, 16> working_set;
    llvm::SmallVector<CallSite, 8> working_set_calls;

    const auto addToWork = [&visited_set, &working_set](auto vals) {
      for (auto v : vals) {
        if (visited_set.find(v) == visited_set.end()) {
          working_set.push_back(v);
          visited_set.insert(v);
        }
      }
    };

    const auto peek = [&working_set]() -> Value* {
      auto user_iter = working_set.end() - 1;
      working_set.erase(user_iter);
      return *user_iter;
    };

    // Seed working set with users of value (e.g., our AllocaInst)
    addToWork(in->users());

    // Search through all users of users of .... (e.g., our AllocaInst)
    while (!working_set.empty()) {
      auto val = peek();

      // If we encounter a callsite, we want to analyze later, or quit in case we have a regex match
      CallSite c(val);
      if (c.isCall()) {
        const auto callee        = c.getCalledFunction();
        const bool indirect_call = callee == nullptr;

        if (indirect_call) {
          LOG_DEBUG("Found an indirect call, not filtering alloca: " << util::dump(*val));
          return false;  // Indirect calls might contain critical function calls.
        }

        const bool is_decl = callee->isDeclaration();
        // FIXME the MPI calls are all hitting this branch (obviously)
        if (is_decl) {
          LOG_DEBUG("Found call with declaration only. Call: " << util::dump(*c.getInstruction()));
          if (c.getIntrinsicID() == Intrinsic::not_intrinsic /*Intrinsic::ID::not_intrinsic*/) {
            if (ClCallFilterDeep && match(callee) && shouldContinue(c, in)) {
              continue;
            }
            return false;
          } else {
            LOG_DEBUG("Call is an intrinsic. Continue analyzing...")
            continue;
          }
        }

        if (match(callee)) {
          LOG_DEBUG("Found a call. Call: " << util::dump(*c.getInstruction()));
          if (ClCallFilterDeep && shouldContinue(c, in)) {
            continue;
          }
          return false;
        }

        working_set_calls.push_back(c);
        // Caveat: below at the end of the loop, we add users of the function call to the search even though it might be
        // a simple "sink" for the alloca we analyse
      } else if (auto store = llvm::dyn_cast<StoreInst>(val)) {
        // If we encounter a store, we follow the store target pointer.
        // More inclusive than strictly necessary in some cases.
        LOG_DEBUG("Store found: " << util::dump(*store)
                                  << " Store target has users: " << util::dump(store->getPointerOperand()->users()));
        auto store_target = store->getPointerOperand();
        // FIXME here we check store operand, if target is another alloca, we already track that?:
        // Note: if we apply this to malloc filtering, this might become problematic?
        if (!malloc_mode && llvm::isa<AllocaInst>(store_target)) {
          LOG_DEBUG("Target is alloca, skipping!");
        } else {
          addToWork(store_target->users());
        }
        continue;
      }
      // cont. our search
      addToWork(val->users());
    }
    ++depth;
    return std::all_of(working_set_calls.begin(), working_set_calls.end(), [&](CallSite c) { return filter(c, in); });
  }

 private:
  bool filter(CallSite& csite, Value* in) {
    const auto analyse_arg = [&](auto& csite, auto argNum) -> bool {
      Argument& the_arg = *(csite.getCalledFunction()->arg_begin() + argNum);
      LOG_DEBUG("Calling filter with inst of argument: " << util::dump(the_arg));
      const bool filter_arg = filter(&the_arg);
      LOG_DEBUG("Should filter? : " << filter_arg);
      return filter_arg;
    };

    LOG_DEBUG("Analyzing function call " << csite.getCalledFunction()->getName());

    if (csite.getCalledFunction() == start_f) {
      return true;
    }

    // this only works if we can correlate alloca with argument:
    const auto pos = std::find_if(csite.arg_begin(), csite.arg_end(),
                                  [&in](const Use& arg_use) -> bool { return arg_use.get() == in; });
    // auto pos = csite.arg_end();
    if (pos != csite.arg_end()) {
      const auto argNum = std::distance(csite.arg_begin(), pos);
      LOG_DEBUG("Found exact position: " << argNum);
      return analyse_arg(csite, argNum);
    } else {
      LOG_DEBUG("Analyze all args, cannot correlate alloca with arg.");
      return std::all_of(csite.arg_begin(), csite.arg_end(), [&csite, &analyse_arg](const Use& arg_use) {
        auto argNum = csite.getArgumentNo(&arg_use);
        return analyse_arg(csite, argNum);
      });
    }

    return true;
  }

  bool filter(Argument* arg) {
    for (auto* user : arg->users()) {
      LOG_DEBUG("Looking at arg user " << util::dump(*user));
      // This code is for non mem2reg code (i.e., where the argument is stored to a local alloca):
      if (auto store = llvm::dyn_cast<StoreInst>(user)) {
        // if (auto* alloca = llvm::dyn_cast<AllocaInst>(store->getPointerOperand())) {
        //  LOG_DEBUG("Argument is a store inst and the operand is alloca");
        return filter(store->getPointerOperand());
        // }
      }
    }
    return filter(llvm::dyn_cast<Value>(arg));
  }

  bool shouldContinue(CallSite c, Value* in) const {
    LOG_DEBUG("Found a name match, analyzing closer...");
    const auto is_void_ptr = [](Type* type) {
      return type->isPointerTy() && type->getPointerElementType()->isIntegerTy(8);
    };
    const auto arg_pos = llvm::find_if(c.args(), [&in](const Use& arg_use) -> bool { return arg_use.get() == in; });
    if (arg_pos == c.arg_end()) {
      // we had no direct correlation for the arg position
      // Now checking if void* is passed, if not we can potentially filter!
      auto count_void_ptr = llvm::count_if(c.args(), [&is_void_ptr](const auto& arg) {
        const auto type = arg->getType();
        return is_void_ptr(type);
      });
      if (count_void_ptr > 0) {
        LOG_DEBUG("Call takes a void*, filtering.");
        return false;
      }
      LOG_DEBUG("Call has no void* argument");
    } else {
      // We have an arg_pos match
      const auto argNum = std::distance(c.arg_begin(), arg_pos);
      Argument& the_arg = *(c.getCalledFunction()->arg_begin() + argNum);
      auto type         = the_arg.getType();
      if (is_void_ptr(type)) {
        LOG_DEBUG("Call arg is a void*, filtering.");
        return false;
      }
      LOG_DEBUG("Value* in is not passed as void ptr");
    }
    LOG_DEBUG("No filter necessary for this call, continue.");
    return true;
  }

  static inline std::string getName(const Function* f) {
    auto name = f->getName();
    // FIXME figure out if we need to demangle, i.e., source is .c or .cpp
    const auto f_name = util::demangle(name);
    if (f_name != "") {
      name = f_name;
    }

    return name;
  }
};

CallFilter::CallFilter(const std::string& glob) : fImpl{std::make_unique<FilterImpl>(glob)} {
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

  mOpsCollector.visit(f);

  LOG_DEBUG("Running on function: " << f.getName())

  const auto checkAmbigiousMalloc = [](const MallocData& mallocData) {
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
          LOG_WARNING("Encountered ambiguous pointer type in allocation: " << util::dump(*(mallocData.call)));
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

const llvm::SmallVectorImpl<GlobalData>& MemInstFinderPass::getModuleGlobals() const {
  return mOpsCollector.globals;
}

}  // namespace typeart

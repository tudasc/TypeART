#include "TypeARTPass.h"
#include "TypeIO.h"
#include "analysis/MemInstFinderPass.h"
#include "support/Logger.h"
#include "support/TypeUtil.h"
#include "support/Util.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Format.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/CtorUtils.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <iostream>
#include <sstream>
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
static cl::opt<bool> ClTypeArtAllocaLifetime("typeart-lifetime",
                                             cl::desc("Track alloca instructions based on lifetime.start intrinsic."),
                                             cl::Hidden, cl::init(false));
static cl::opt<std::string> ClTypeFile("typeart-outfile", cl::desc("Location of the generated type file."), cl::Hidden,
                                       cl::init("types.yaml"));

// FIXME 1) include bitcasts? 2) disabled by default in LLVM builds (use LLVM_ENABLE_STATS when building)
// STATISTIC(NumInstrumentedMallocs, "Number of instrumented mallocs");
// STATISTIC(NumInstrumentedFrees, "Number of instrumented frees");
STATISTIC(NumFoundMallocs, "Number of detected mallocs");
STATISTIC(NumFoundFrees, "Number of detected frees");
STATISTIC(NumFoundAlloca, "Number of detected (stack) allocas");

namespace tu = typeart::util::type;

namespace typeart {
namespace pass {

// Used by LLVM pass manager to identify passes in memory
char TypeArtPass::ID = 0;

// std::unique_ptr<TypeMapping> TypeArtPass::typeMapping = std::make_unique<SimpleTypeMapping>();

class FilterImpl {
  const std::string call_regex;
  bool malloc_mode{false};
  int depth{0};

 public:
  explicit FilterImpl(const std::string& glob) : call_regex(util::glob2regex(glob)) {
  }

  void setMode(bool search_malloc) {
    malloc_mode = search_malloc;
  }

  bool filter(Value* in) {
    if (in == nullptr) {
      LOG_DEBUG("Called with nullptr");
      return false;
    }
    if (depth == 15)
      return false;

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
        const auto callee = c.getCalledFunction();
        const bool indirect_call = callee == nullptr;

        if (indirect_call) {
          LOG_DEBUG("Found an indirect call, not filtering alloca: " << util::dump(*val));
          return false;  // Indirect calls might contain critical function calls.
        }

        const bool is_decl = callee->isDeclaration();
        // FIXME the MPI calls are all hitting this branch (obviously)
        if (is_decl) {
          LOG_DEBUG("Found call with declaration only. Call: " << util::dump(*c.getInstruction()));
          if (c.getIntrinsicID() == Intrinsic::ID::not_intrinsic) {
            if (match(callee) && shouldContinue(c, in)) {
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
          if (shouldContinue(c, in)) {
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
      auto type = the_arg.getType();
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

TypeArtPass::TypeArtPass() : llvm::FunctionPass(ID), typeManager(ClTypeFile.getValue()) {
  assert(!ClTypeFile.empty() && "Default type file not set");
}

void TypeArtPass::getAnalysisUsage(llvm::AnalysisUsage& info) const {
  info.addRequired<typeart::MemInstFinderPass>();
}

bool TypeArtPass::doInitialization(Module& m) {
  /**
   * Introduce the necessary instrumentation functions in the LLVM module.
   * functions:
   * void __typeart_alloc(void *addr, int typeId, size_t count, size_t typeSize, int isLocal)
   * void __typeart_free(void *ptr)
   *
   * Also scan the LLVM module for type definitions and add them to our type list.
   */

  propagateTypeInformation(m);

  if (!ClIgnoreHeap) {
    declareInstrumentationFunctions(m);

    // Find globals
    LOG_DEBUG("Collecting global variables in module " << m.getSourceFileName() << "...");
    DataLayout dl(&m);

    auto& c = m.getContext();

    const auto shouldInstrumentGlobal = [&](GlobalVariable& g) -> bool {
      if (g.getName().startswith("llvm.")) {
        // LOG_DEBUG("Detected LLVM global " << global.getName() << " - skipping...");
        return false;
      }

      // TODO: Filter based on linkage types? (see address sanitizer)

      Type* t = g.getValueType();
      if (!t->isSized()) {
        return false;
      }

      if (t->isArrayTy()) {
        t = t->getArrayElementType();
      }
      if (auto structType = dyn_cast<StructType>(t)) {
        if (structType->isOpaque()) {
          LOG_DEBUG("Encountered opaque struct " << t->getStructName() << " - skipping...");
          return false;
        }
      }

      FilterImpl filter("MPI_*");
      return !filter.filter(&g);
    };

    const auto instrumentGlobal = [&](auto* global, auto& IRB) {
      auto type = global->getValueType();

      unsigned numElements = 1;
      if (type->isArrayTy()) {
        numElements = tu::getArrayLengthFlattened(type);
        type = tu::getArrayElementType(type);
      }

      int typeId = typeManager.getOrRegisterType(type, dl);
      unsigned typeSize = tu::getTypeSizeInBytes(type, dl);

      auto typeIdConst = ConstantInt::get(tu::getInt32Type(c), typeId);
      auto typeSizeConst = ConstantInt::get(tu::getInt64Type(c), typeSize);
      auto numElementsConst = ConstantInt::get(tu::getInt64Type(c), numElements);
      auto isLocalConst = ConstantInt::get(tu::getInt32Type(c), 0);

      auto globalPtr = IRB.CreateBitOrPointerCast(global, tu::getVoidPtrType(c));

      LOG_DEBUG("Instrumenting global variable: " << util::dump(*globalPtr));

      IRB.CreateCall(typeart_alloc.f,
                     ArrayRef<Value*>{globalPtr, typeIdConst, numElementsConst, typeSizeConst, isLocalConst});
    };

    SmallVector<GlobalVariable*, 8> globalsList;
    std::for_each(m.global_begin(), m.global_end(), [&](auto& global) {
      if (shouldInstrumentGlobal(global)) {
        globalsList.push_back(&global);
      }
    });

    if (!globalsList.empty()) {
      auto ctorFunctionName = "__typeart_init_module_" + m.getSourceFileName();

      FunctionType* ctorType = FunctionType::get(llvm::Type::getVoidTy(c), false);
      Function* ctorFunction = Function::Create(ctorType, Function::PrivateLinkage, ctorFunctionName, &m);

      BasicBlock* entry = BasicBlock::Create(c, "entry", ctorFunction);
      IRBuilder<> IRB(entry);

      using namespace std::placeholders;
      std::for_each(globalsList.begin(), globalsList.end(), std::bind(instrumentGlobal, _1, std::ref(IRB)));

      IRB.CreateRetVoid();

      llvm::appendToGlobalCtors(m, ctorFunction, 0, nullptr);
    }
  }

  return true;
}

bool TypeArtPass::runOnFunction(Function& f) {
  using namespace typeart;

  // Ignore our own functions
  if (f.getName().startswith("__typeart")) {
    return false;
  }

  LOG_DEBUG("Running on function: " << f.getName())

  // FIXME this is required when "PassManagerBuilder::EP_OptimizerLast" is used as the function (constant) pointer are
  // nullpointer/invalidated
  declareInstrumentationFunctions(*f.getParent());

  bool mod{false};
  auto& c = f.getContext();
  DataLayout dl(f.getParent());

  llvm::SmallDenseMap<BasicBlock*, size_t> allocCounts;

  const auto& listMalloc = getAnalysis<MemInstFinderPass>().getFunctionMallocs();
  const auto& listAlloca = getAnalysis<MemInstFinderPass>().getFunctionAllocs();
  const auto& listFree = getAnalysis<MemInstFinderPass>().getFunctionFrees();

  const auto instrumentMalloc = [&](const auto& malloc) -> bool {
    const auto mallocInst = malloc.call;
    BitCastInst* primaryBitcast = malloc.primary;

    // Number of bytes allocated
    auto mallocArg = mallocInst->getOperand(0);
    int typeId = typeManager.getOrRegisterType(mallocInst->getType()->getPointerElementType(),
                                               dl);  // retrieveTypeID(tu::getVoidType(c));
    // Number of bytes per element, 1 for void*
    unsigned typeSize = tu::getTypeSizeInBytes(mallocInst->getType()->getPointerElementType(), dl);
    auto insertBefore = mallocInst->getNextNode();

    // Use the first cast as the determining type (if there is any)
    if (primaryBitcast) {
      auto dstPtrType = primaryBitcast->getDestTy()->getPointerElementType();
      typeSize = tu::getTypeSizeInBytes(dstPtrType, dl);
      typeId = typeManager.getOrRegisterType(dstPtrType, dl);  //(unsigned)dstPtrType->getTypeID();
    }

    IRBuilder<> IRB(insertBefore);
    auto typeIdConst = ConstantInt::get(tu::getInt32Type(c), typeId);
    auto typeSizeConst = ConstantInt::get(tu::getInt64Type(c), typeSize);
    // Compute element count: count = numBytes / typeSize
    Value* elementCount = nullptr;
    if (malloc.kind == MemOpKind::MALLOC) {
      elementCount = IRB.CreateUDiv(mallocArg, typeSizeConst);
    } else if (malloc.kind == MemOpKind::CALLOC) {
      elementCount = mallocInst->getOperand(0);  // get the element count in calloc call
    } else if (malloc.kind == MemOpKind::REALLOC) {
      auto mArg = mallocInst->getOperand(1);
      elementCount = IRB.CreateUDiv(mArg, typeSizeConst);

      IRBuilder<> FreeB(mallocInst);
      auto addrOp = mallocInst->getOperand(0);
      FreeB.CreateCall(typeart_free.f, ArrayRef<Value*>{addrOp});

    } else {
      LOG_ERROR("Unknown malloc kind. Not instrumenting. " << util::dump(*mallocInst));
      return false;
    }

    auto isLocalConst = ConstantInt::get(tu::getInt32Type(c), 0);
    IRB.CreateCall(typeart_alloc.f,
                   ArrayRef<Value*>{mallocInst, typeIdConst, elementCount, typeSizeConst, isLocalConst});

    return true;
  };

  const auto instrumentFree = [&](const auto& free) -> bool {
    // Pointer address:
    auto freeArg = free->getOperand(0);
    IRBuilder<> IRB(free->getNextNode());
    IRB.CreateCall(typeart_free.f, ArrayRef<Value*>{freeArg});

    return true;
  };

  const auto instrumentAlloca = [&](const auto& allocaData) -> bool {
    auto alloca = allocaData.alloca;

    Type* elementType = alloca->getAllocatedType();
    unsigned arraySize = 1;

    if (elementType->isArrayTy()) {
      arraySize = tu::getArrayLengthFlattened(elementType);
      elementType = tu::getArrayElementType(elementType);
    }

    unsigned typeSize = tu::getTypeSizeInBytes(elementType, dl);

    int typeId = typeManager.getOrRegisterType(elementType, dl);
    auto typeIdConst = ConstantInt::get(tu::getInt32Type(c), typeId);
    auto typeSizeConst = ConstantInt::get(tu::getInt64Type(c), typeSize);
    auto numElementsConst = ConstantInt::get(tu::getInt64Type(c), arraySize);
    auto isLocalConst = ConstantInt::get(tu::getInt32Type(c), 1);

    if (ClTypeArtAllocaLifetime) {
      // TODO Using lifetime start (and end) likely cause our counter based stack tracking scheme to fail?
      auto marker = allocaData.start;
      if (marker != nullptr) {
        IRBuilder<> IRB(marker->getNextNode());

        auto arrayPtr = marker->getOperand(1);  // IRB.CreateBitOrPointerCast(alloca, tu::getVoidPtrType(c));
        // LOG_DEBUG("Using lifetime marker for alloca: " << util::dump(*arrayPtr));
        IRB.CreateCall(typeart_alloc.f,
                       ArrayRef<Value*>{arrayPtr, typeIdConst, numElementsConst, typeSizeConst, isLocalConst});

        allocCounts[marker->getParent()]++;

        ++NumFoundAlloca;
        return true;
      }
    }

    IRBuilder<> IRB(alloca->getNextNode());

    // Single increment for an alloca:
    //    auto load_counter = IRB.CreateLoad(counter);
    //    Value* increment_counter = IRB.CreateAdd(IRB.getInt64(1), load_counter);
    //    IRB.CreateStore(increment_counter, counter);

    auto arrayPtr = IRB.CreateBitOrPointerCast(alloca, tu::getVoidPtrType(c));
    IRB.CreateCall(typeart_alloc.f,
                   ArrayRef<Value*>{arrayPtr, typeIdConst, numElementsConst, typeSizeConst, isLocalConst});

    allocCounts[alloca->getParent()]++;

    ++NumFoundAlloca;
    return true;
  };

  if (!ClIgnoreHeap) {
    // instrument collected calls of bb:
    for (auto& malloc : listMalloc) {
      ++NumFoundMallocs;
      mod |= instrumentMalloc(malloc);
    }

    for (auto free : listFree) {
      ++NumFoundFrees;
      mod |= instrumentFree(free);
    }
  }
  if (ClTypeArtAlloca) {
    const bool instrumented_alloca = std::count_if(listAlloca.begin(), listAlloca.end(), instrumentAlloca) > 0;
    mod |= instrumented_alloca;

    if (instrumented_alloca) {
      //      LOG_DEBUG("Add alloca counter")
      // counter = 0 at beginning of function
      IRBuilder<> CBuilder(f.getEntryBlock().getFirstNonPHI());
      auto counter = CBuilder.CreateAlloca(tu::getInt64Type(c), nullptr, "__ta_alloca_counter");
      CBuilder.CreateStore(CBuilder.getInt64(0), counter);

      // In each basic block: counter =+ num_alloca (in BB)
      for (auto data : allocCounts) {
        IRBuilder<> IRB(data.first->getTerminator());
        auto load_counter = IRB.CreateLoad(counter);
        Value* increment_counter = IRB.CreateAdd(IRB.getInt64(data.second), load_counter);
        IRB.CreateStore(increment_counter, counter);
      }

      // Find return instructions:
      // if(counter > 0) call runtime for stack cleanup
      EscapeEnumerator ee(f);
      while (IRBuilder<>* irb = ee.Next()) {
        auto I = &(*irb->GetInsertPoint());

        auto counter_load = irb->CreateLoad(counter, "__ta_counter_load");
        auto cond = irb->CreateICmpNE(counter_load, irb->getInt64(0), "__ta_cond");
        auto then_term = SplitBlockAndInsertIfThen(cond, I, false);
        irb->SetInsertPoint(then_term);
        irb->CreateCall(typeart_leave_scope.f, ArrayRef<Value*>{counter_load});
      }
    }
  } else {
    NumFoundAlloca += listAlloca.size();
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
    LOG_ERROR("Failed writing type config to " << ClTypeFile.getValue());
  }
  if (ClTypeArtStats) {
    printStats(llvm::errs());
  }
  return false;
}

void TypeArtPass::declareInstrumentationFunctions(Module& m) {
  // Remove this return if problems come up during compilation
  if (typeart_alloc.f != nullptr && typeart_free.f != nullptr && typeart_leave_scope.f != nullptr) {
    return;
  }

  const auto addOptimizerAttributes = [&](auto& arg) {
    arg.addAttr(Attribute::NoCapture);
    arg.addAttr(Attribute::ReadOnly);
  };

  const auto make_function = [&](auto& f_struct, auto f_type) {
    f_struct.f = m.getOrInsertFunction(f_struct.name, f_type);
    if (auto f = dyn_cast<Function>(f_struct.f)) {
      f->setLinkage(GlobalValue::ExternalLinkage);
      auto& firstParam = *(f->arg_begin());
      if (firstParam.getType()->isPointerTy()) {
        addOptimizerAttributes(firstParam);
      }
    }
  };

  auto& c = m.getContext();
  Type* alloc_arg_types[] = {tu::getVoidPtrType(c), tu::getInt32Type(c), tu::getInt64Type(c), tu::getInt64Type(c),
                             tu::getInt32Type(c)};
  Type* free_arg_types[] = {tu::getVoidPtrType(c)};

  make_function(typeart_alloc, FunctionType::get(Type::getVoidTy(c), alloc_arg_types, false));
  make_function(typeart_free, FunctionType::get(Type::getVoidTy(c), free_arg_types, false));
  make_function(typeart_leave_scope,
                FunctionType::get(Type::getVoidTy(c), ArrayRef<Type*>{tu::getInt64Type(c)}, false));
}

void TypeArtPass::propagateTypeInformation(Module&) {
  /* Read already acquired information from temporary storage */
  /*
   * Scan module for type definitions and add to the type information map
   * Type information needed:
   *  + Name
   *  + Data member
   *  + Extent
   *  + Our id
   */
  LOG_DEBUG("Propagating type infos.");
  if (typeManager.load()) {
    LOG_DEBUG("Existing type configuration successfully loaded from " << ClTypeFile.getValue());
  } else {
    LOG_DEBUG("No valid existing type configuration found: " << ClTypeFile.getValue());
  }
}

void TypeArtPass::printStats(llvm::raw_ostream& out) {
  const unsigned max_string{12u};
  const unsigned max_val{5u};
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
  out << make_format("Malloc", NumFoundMallocs.getValue());
  out << make_format("Free", NumFoundFrees.getValue());
  out << line;
  out << "Stack Memory\n";
  out << line;
  out << make_format("Alloca", NumFoundAlloca.getValue());
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

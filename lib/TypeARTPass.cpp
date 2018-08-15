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

STATISTIC(NumInstrumentedMallocs, "Number of instrumented mallocs");
STATISTIC(NumInstrumentedFrees, "Number of instrumented frees");
STATISTIC(NumInstrumentedAlloca, "Number of instrumented (stack) allocas");
STATISTIC(NumInstrumentedGlobal, "Number of instrumented globals");

namespace tu = typeart::util::type;

namespace typeart {
namespace pass {

// Used by LLVM pass manager to identify passes in memory
char TypeArtPass::ID = 0;

// std::unique_ptr<TypeMapping> TypeArtPass::typeMapping = std::make_unique<SimpleTypeMapping>();

TypeArtPass::TypeArtPass() : llvm::ModulePass(ID), typeManager(ClTypeFile.getValue()) {
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

  return true;
}

bool TypeArtPass::runOnModule(Module& m) {
  bool globalInstro{false};
  if (ClIgnoreHeap) {
    declareInstrumentationFunctions(m);

    DataLayout dl(&m);

    auto& c = m.getContext();

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
      auto isLocalConst = ConstantInt::get(tu::getInt32Type(c), 2);

      auto globalPtr = IRB.CreateBitOrPointerCast(global, tu::getVoidPtrType(c));

      LOG_DEBUG("Instrumenting global variable: " << util::dump(*global));

      IRB.CreateCall(typeart_alloc_global.f,
                     ArrayRef<Value*>{globalPtr, typeIdConst, numElementsConst, typeSizeConst, isLocalConst});
      return true;
    };

    const auto makeCtorFunc = [&]() -> IRBuilder<> {
      auto ctorFunctionName = "__typeart_init_module_" + m.getSourceFileName();

      FunctionType* ctorType = FunctionType::get(llvm::Type::getVoidTy(c), false);
      Function* ctorFunction = Function::Create(ctorType, Function::PrivateLinkage, ctorFunctionName, &m);

      BasicBlock* entry = BasicBlock::Create(c, "entry", ctorFunction);

      llvm::appendToGlobalCtors(m, ctorFunction, 0, nullptr);

      IRBuilder<> IRB(entry);
      return IRB;
    };

    const auto& globalsList = getAnalysis<MemInstFinderPass>().getModuleGlobals();
    if (!globalsList.empty()) {
      auto IRB = makeCtorFunc();
      auto instrGlobalCount = llvm::count_if(globalsList, [&](auto g) { return instrumentGlobal(g, IRB); });
      NumInstrumentedGlobal += instrGlobalCount;
      globalInstro = instrGlobalCount > 0;
      IRB.CreateRetVoid();
    }
  }
  const auto instrumentedF = llvm::count_if(m.functions(), [&](auto& f) { return runOnFunc(f); }) > 0;
  return instrumentedF || globalInstro;
}

bool TypeArtPass::runOnFunc(Function& f) {
  using namespace typeart;

  if (f.isDeclaration() || f.getName().startswith("__typeart")) {
    return false;
  }

  if (!getAnalysis<MemInstFinderPass>().hasFunctionData(&f)) {
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

  const auto& fData = getAnalysis<MemInstFinderPass>().getFunctionData(&f);
  const auto& listMalloc = fData.listMalloc;
  const auto& listAlloca = fData.listAlloca;
  const auto& listFree = fData.listFree;

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

    IRB.CreateCall(typeart_alloc.f, ArrayRef<Value*>{mallocInst, typeIdConst, elementCount, typeSizeConst});

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
        IRB.CreateCall(typeart_alloc_stack.f, ArrayRef<Value*>{arrayPtr, typeIdConst, numElementsConst, typeSizeConst});

        allocCounts[marker->getParent()]++;

        ++NumInstrumentedAlloca;
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

    ++NumInstrumentedAlloca;
    return true;
  };

  if (!ClIgnoreHeap) {
    // instrument collected calls of bb:
    for (auto& malloc : listMalloc) {
      ++NumInstrumentedMallocs;
      mod |= instrumentMalloc(malloc);
    }

    for (auto free : listFree) {
      ++NumInstrumentedFrees;
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
    NumInstrumentedAlloca += listAlloca.size();
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
  if (typeart_alloc_global.f != nullptr && typeart_alloc_stack.f != nullptr && typeart_alloc.f != nullptr &&
      typeart_free.f != nullptr && typeart_leave_scope.f != nullptr) {
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
  Type* alloc_arg_types[] = {tu::getVoidPtrType(c), tu::getInt32Type(c), tu::getInt64Type(c), tu::getInt64Type(c)};
  Type* free_arg_types[] = {tu::getVoidPtrType(c)};
  Type* leavescope_arg_types[] = {tu::getInt64Type(c)};

  make_function(typeart_alloc, FunctionType::get(Type::getVoidTy(c), alloc_arg_types, false));
  make_function(typeart_alloc_stack, FunctionType::get(Type::getVoidTy(c), alloc_arg_types, false));
  make_function(typeart_alloc_global, FunctionType::get(Type::getVoidTy(c), alloc_arg_types, false));
  make_function(typeart_free, FunctionType::get(Type::getVoidTy(c), free_arg_types, false));
  make_function(typeart_leave_scope, FunctionType::get(Type::getVoidTy(c), leavescope_arg_types, false));
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

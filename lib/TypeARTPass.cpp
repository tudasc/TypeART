#include "TypeARTPass.h"

#include "RuntimeInterface.h"
#include "TypeIO.h"
#include "TypeInterface.h"
#include "analysis/MemInstFinderPass.h"
#include "support/Logger.h"
#include "support/TypeUtil.h"
#include "support/Util.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
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
  EnableStatistics();
}

void TypeArtPass::getAnalysisUsage(llvm::AnalysisUsage& info) const {
  info.addRequired<typeart::MemInstFinderPass>();
}

bool TypeArtPass::doInitialization(Module& m) {
  instr.setModule(m);

  LOG_DEBUG("Propagating type infos.");
  if (typeManager.load()) {
    LOG_DEBUG("Existing type configuration successfully loaded from " << ClTypeFile.getValue());
  } else {
    LOG_DEBUG("No valid existing type configuration found: " << ClTypeFile.getValue());
  }

  return true;
}

bool TypeArtPass::runOnModule(Module& m) {
  bool globalInstro{false};
  if (ClIgnoreHeap) {
    declareInstrumentationFunctions(m);

    DataLayout dl(&m);

    auto& c = m.getContext();

    const auto instrumentGlobal = [&](auto& global_data, auto& IRB) {
      auto global = global_data.global;
      auto type   = global->getValueType();

      unsigned numElements = 1;
      if (type->isArrayTy()) {
        numElements = tu::getArrayLengthFlattened(type);
        type        = tu::getArrayElementType(type);
      }

      int typeId             = typeManager.getOrRegisterType(type, dl);
      auto* typeIdConst      = instr.getConstantFor(IType::type_id, typeId);
      auto* numElementsConst = instr.getConstantFor(IType::extent, numElements);
      auto globalPtr         = IRB.CreateBitOrPointerCast(global, instr.getTypeFor(IType::ptr));

      LOG_DEBUG("Instrumenting global variable: " << util::dump(*global));

      IRB.CreateCall(typeart_alloc_global.f, ArrayRef<Value*>{globalPtr, typeIdConst, numElementsConst});
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
      auto IRB              = makeCtorFunc();
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

  const auto instrumentMalloc = [&](const auto& malloc) -> bool {
    const auto malloc_call      = malloc.call;
    BitCastInst* primaryBitcast = malloc.primary;

    // Number of bytes allocated
    auto mallocArg = malloc_call->getOperand(0);
    int typeId     = typeManager.getOrRegisterType(malloc_call->getType()->getPointerElementType(),
                                               dl);  // retrieveTypeID(tu::getVoidType(c));
    if (typeId == TA_UNKNOWN_TYPE) {
      LOG_ERROR("Unknown allocated type. Not instrumenting. " << util::dump(*malloc_call));
      return false;
    }

    // Number of bytes per element, 1 for void*
    unsigned typeSize = tu::getTypeSizeInBytes(malloc_call->getType()->getPointerElementType(), dl);
    auto insertBefore = malloc_call->getNextNode();

    // Use the first cast as the determining type (if there is any)
    if (primaryBitcast) {
      auto* dstPtrType = primaryBitcast->getDestTy()->getPointerElementType();

      typeSize = tu::getTypeSizeInBytes(dstPtrType, dl);

      // Resolve arrays
      // TODO: Write tests for this case
      if (dstPtrType->isArrayTy()) {
        dstPtrType = tu::getArrayElementType(dstPtrType);
      }

      typeId = typeManager.getOrRegisterType(dstPtrType, dl);
      if (typeId == TA_UNKNOWN_TYPE) {
        LOG_ERROR("Target type of casted allocation is unknown. Not instrumenting. " << util::dump(*malloc_call));
        LOG_ERROR("Cast: " << util::dump(*primaryBitcast));
        LOG_ERROR("Target type: " << util::dump(*dstPtrType));
        return false;
      }
    } else {
      LOG_ERROR("Primary bitcast is null. malloc: " << util::dump(*malloc_call))
    }

    if (malloc.is_invoke) {
      InvokeInst* inv = dyn_cast<InvokeInst>(malloc_call);
      insertBefore    = &(*inv->getNormalDest()->getFirstInsertionPt());
    }

    IRBuilder<> IRB(insertBefore);
    auto* typeIdConst   = instr.getConstantFor(IType::type_id, typeId);
    auto* typeSizeConst = instr.getConstantFor(IType::extent, typeSize);
    // Compute element count: count = numBytes / typeSize
    Value* elementCount = nullptr;
    if (malloc.kind == MemOpKind::MALLOC) {
      elementCount = IRB.CreateUDiv(mallocArg, typeSizeConst);
    } else if (malloc.kind == MemOpKind::CALLOC) {
      elementCount = malloc_call->getOperand(0);  // get the element count in calloc call
    } else if (malloc.kind == MemOpKind::REALLOC) {
      auto mArg    = malloc_call->getOperand(1);
      elementCount = IRB.CreateUDiv(mArg, typeSizeConst);

      IRBuilder<> FreeB(malloc_call);
      auto addrOp = malloc_call->getOperand(0);
      FreeB.CreateCall(typeart_free.f, ArrayRef<Value*>{addrOp});

    } else {
      LOG_ERROR("Unknown malloc kind. Not instrumenting. " << util::dump(*malloc_call));
      return false;
    }

    IRB.CreateCall(typeart_alloc.f, ArrayRef<Value*>{malloc_call, typeIdConst, elementCount});

    return true;
  };

  const auto instrumentFree = [&](const auto& free_data) -> bool {
    auto free_call    = free_data.call;
    auto freeArg      = free_call->getOperand(0);
    auto insertBefore = free_call->getNextNode();
    if (free_data.is_invoke) {
      InvokeInst* inv = dyn_cast<InvokeInst>(free_call);
      insertBefore    = &(*inv->getNormalDest()->getFirstInsertionPt());
    }
    IRBuilder<> IRB(insertBefore);
    IRB.CreateCall(typeart_free.f, ArrayRef<Value*>{freeArg});

    return true;
  };

  const auto instrumentAlloca = [&](const auto& allocaData) -> bool {
    auto alloca           = allocaData.alloca;
    Type* elementType     = alloca->getAllocatedType();
    Value* numElementsVal = nullptr;
    // The length can be specified statically through the array type or as a separate argument.
    // Both cases are handled here.
    if (allocaData.is_vla) {
      numElementsVal = alloca->getArraySize();
      // This should not happen in generated IR code
      assert(!elementType->isArrayTy() && "VLAs of array types are currently not supported.");
    } else {
      size_t arraySize = allocaData.array_size;
      if (elementType->isArrayTy()) {
        arraySize   = arraySize * tu::getArrayLengthFlattened(elementType);
        elementType = tu::getArrayElementType(elementType);
      }
      numElementsVal = instr.getConstantFor(IType::extent, arraySize);
    }

    IRBuilder<> IRB(alloca->getNextNode());

    // unsigned typeSize = tu::getTypeSizeInBytes(elementType, dl);
    int typeId = typeManager.getOrRegisterType(elementType, dl);

    if (typeId == TA_UNKNOWN_TYPE) {
      LOG_ERROR("Type is not supported: " << util::dump(*elementType));
    }

    auto* typeIdConst = instr.getConstantFor(IType::type_id, typeId);
    auto arrayPtr     = IRB.CreateBitOrPointerCast(alloca, instr.getTypeFor(IType::ptr));

    IRB.CreateCall(typeart_alloc_stack.f, ArrayRef<Value*>{arrayPtr, typeIdConst, numElementsVal});

    allocCounts[alloca->getParent()]++;

    ++NumInstrumentedAlloca;
    return true;
  };

  if (!ClIgnoreHeap) {
    // instrument collected calls of bb:
    for (const auto& malloc : mallocs) {
      ++NumInstrumentedMallocs;
      mod |= instrumentMalloc(malloc);
    }

    for (auto& free : frees) {
      ++NumInstrumentedFrees;
      mod |= instrumentFree(free);
    }
  }
  if (ClTypeArtAlloca) {
    const bool instrumented_alloca = std::count_if(allocas.begin(), allocas.end(), instrumentAlloca) > 0;
    mod |= instrumented_alloca;

    if (instrumented_alloca) {
      //      LOG_DEBUG("Add alloca counter")
      // counter = 0 at beginning of function
      IRBuilder<> CBuilder(f.getEntryBlock().getFirstNonPHI());
      auto* counter = CBuilder.CreateAlloca(instr.getTypeFor(IType::stack_count), nullptr, "__ta_alloca_counter");
      CBuilder.CreateStore(instr.getConstantFor(IType::stack_count), counter);

      // In each basic block: counter =+ num_alloca (in BB)
      for (auto data : allocCounts) {
        IRBuilder<> IRB(data.first->getTerminator());
        auto* load_counter       = IRB.CreateLoad(counter);
        Value* increment_counter = IRB.CreateAdd(instr.getConstantFor(IType::stack_count, data.second), load_counter);
        IRB.CreateStore(increment_counter, counter);
      }

      // Find return instructions:
      // if(counter > 0) call runtime for stack cleanup
      EscapeEnumerator ee(f);
      while (IRBuilder<>* irb = ee.Next()) {
        auto* I = &(*irb->GetInsertPoint());

        auto* counter_load = irb->CreateLoad(counter, "__ta_counter_load");
        auto* cond         = irb->CreateICmpNE(counter_load, instr.getConstantFor(IType::stack_count), "__ta_cond");
        auto* then_term    = SplitBlockAndInsertIfThen(cond, I, false);
        irb->SetInsertPoint(then_term);
        irb->CreateCall(typeart_leave_scope.f, ArrayRef<Value*>{counter_load});
      }
    }
  } else {
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

  auto alloc_arg_types      = instr.make_parameters(IType::ptr, IType::type_id, IType::extent);
  auto free_arg_types       = instr.make_parameters(IType::ptr);
  auto leavescope_arg_types = instr.make_parameters(IType::stack_count);

  typeart_alloc.f        = instr.make_function(typeart_alloc.name, alloc_arg_types);
  typeart_alloc_stack.f  = instr.make_function(typeart_alloc_stack.name, alloc_arg_types);
  typeart_alloc_global.f = instr.make_function(typeart_alloc_global.name, alloc_arg_types);
  typeart_free.f         = instr.make_function(typeart_free.name, free_arg_types);
  typeart_leave_scope.f  = instr.make_function(typeart_leave_scope.name, leavescope_arg_types);
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

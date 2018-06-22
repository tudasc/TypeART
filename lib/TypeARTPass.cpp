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
#include "llvm/Transforms/Utils/EscapeEnumerator.h"

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
static cl::opt<bool> ClTypeArtAlloca("typeart-alloca", cl::desc("Track alloca instructions."), cl::Hidden,
                                     cl::init(false));
static cl::opt<std::string> ClTypeFile("typeart-outfile", cl::desc("Location of the generated type file."), cl::Hidden,
                                       cl::init("types.yaml"));

// FIXME 1) include bitcasts? 2) disabled by default in LLVM builds (use LLVM_ENABLE_STATS when building)
// STATISTIC(NumInstrumentedMallocs, "Number of instrumented mallocs");
// STATISTIC(NumInstrumentedFrees, "Number of instrumented frees");
STATISTIC(NumFoundMallocs, "Number of detected mallocs");
STATISTIC(NumFoundFrees, "Number of detected frees");
STATISTIC(NumFoundAlloca, "Number of detected (stack) allocas");

namespace tu = util::type;

namespace typeart {
namespace pass {

// Used by LLVM pass manager to identify passes in memory
char TypeArtPass::ID = 0;

// std::unique_ptr<TypeMapping> TypeArtPass::typeMapping = std::make_unique<SimpleTypeMapping>();

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
  declareInstrumentationFunctions(m);

  propagateTypeInformation(m);

  return true;
}

bool TypeArtPass::runOnFunction(Function& f) {
  using namespace typeart;

  bool mod{false};
  auto& c = f.getContext();
  DataLayout dl(f.getParent());

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
      insertBefore = primaryBitcast->getNextNode();
    }

    IRBuilder<> IRB(insertBefore);
    auto typeIdConst = ConstantInt::get(tu::getInt32Type(c), typeId);
    auto typeSizeConst = ConstantInt::get(tu::getInt64Type(c), typeSize);
    // Compute element count: count = numBytes / typeSize
    auto elementCount = IRB.CreateUDiv(mallocArg, typeSizeConst);
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

  const auto instrumentAlloca = [&](const auto& alloca) -> bool {
    unsigned typeSize = tu::getTypeSizeForArrayAlloc(alloca, dl);
    auto insertBefore = alloca->getNextNode();

    auto elementType = alloca->getAllocatedType()->getArrayElementType();

    int typeId = typeManager.getOrRegisterType(elementType, dl);
    auto arraySize = alloca->getAllocatedType()->getArrayNumElements();

    auto typeIdConst = ConstantInt::get(tu::getInt32Type(c), typeId);
    auto typeSizeConst = ConstantInt::get(tu::getInt64Type(c), typeSize);
    auto numElementsConst = ConstantInt::get(tu::getInt64Type(c), arraySize);
    auto isLocalConst = ConstantInt::get(tu::getInt32Type(c), 1);

    IRBuilder<> IRB(insertBefore);
    auto arrayPtr = IRB.CreateBitOrPointerCast(alloca, tu::getVoidPtrType(c));
    IRB.CreateCall(typeart_alloc.f,
                   ArrayRef<Value*>{arrayPtr, typeIdConst, numElementsConst, typeSizeConst, isLocalConst});

    return true;
  };

  const auto instrumentEnterScope = [&](auto& insertBefore) -> bool {
    IRBuilder<> IRB(&insertBefore);
    IRB.CreateCall(typeart_enter_scope.f);
    return true;
  };

  const auto instrumentLeaveScope = [&](auto& builder) -> bool {
    builder.CreateCall(typeart_leave_scope.f);
    return true;
  };

  // instrument collected calls of bb:
  for (auto& malloc : listMalloc) {
    ++NumFoundMallocs;
    mod |= instrumentMalloc(malloc);
  }

  for (auto free : listFree) {
    ++NumFoundFrees;
    mod |= instrumentFree(free);
  }

  if (ClTypeArtAlloca) {
    // Create new scope
    auto& entryPoint = *f.getEntryBlock().getFirstInsertionPt();
    instrumentEnterScope(entryPoint);

    // Find return instructions and exceptions
    EscapeEnumerator ee(f);
    while (IRBuilder<>* beforeEscape = ee.Next()) {
      instrumentLeaveScope(*beforeEscape);
    }

    for (auto alloca : listAlloca) {
      if (alloca->getAllocatedType()->isArrayTy()) {
        ++NumFoundAlloca;
        mod |= instrumentAlloca(alloca);
      }
    }
  } else {
    // FIXME just for counting (and make tests pass)
    for (auto alloca : listAlloca) {
      if (alloca->getAllocatedType()->isArrayTy()) {
        ++NumFoundAlloca;
      }
    }
  }

  return mod;
}

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
  const auto make_function = [&m](auto& f_struct, auto f_type) {
    f_struct.f = m.getOrInsertFunction(f_struct.name, f_type);
    if (auto f = dyn_cast<Function>(f_struct.f)) {
      f->setLinkage(GlobalValue::ExternalLinkage);
    }
  };

  auto& c = m.getContext();
  Type* alloc_arg_types[] = {tu::getVoidPtrType(c), tu::getInt32Type(c), tu::getInt64Type(c), tu::getInt64Type(c),
                             tu::getInt32Type(c)};
  Type* free_arg_types[] = {tu::getVoidPtrType(c)};

  make_function(typeart_alloc, FunctionType::get(Type::getVoidTy(c), alloc_arg_types, false));
  make_function(typeart_free, FunctionType::get(Type::getVoidTy(c), free_arg_types, false));
  make_function(typeart_enter_scope, FunctionType::get(Type::getVoidTy(c), false));
  make_function(typeart_leave_scope, FunctionType::get(Type::getVoidTy(c), false));
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
static RegisterStandardPasses RegisterClangPass(PassManagerBuilder::EP_EarlyAsPossible, registerClangPass);

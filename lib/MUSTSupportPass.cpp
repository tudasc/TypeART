#include "MUSTSupportPass.h"
#include "MemOpVisitor.h"
#include "support/Logger.h"
#include "support/TypeUtil.h"
#include "support/Util.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Format.h"

#include <TypeIO.h>
#include <iostream>
#include <sstream>
#include <string>

using namespace llvm;

#define DEBUG_TYPE "must"

namespace {
static llvm::RegisterPass<must::pass::MustSupportPass> msp("must", "MUST type information", false, false);
}  // namespace

static cl::opt<bool> ClMustStats("must-stats", cl::desc("Show statistics for MUST type pass."), cl::Hidden,
                                 cl::init(false));

static cl::opt<std::string> ClConfigDir("config-dir", cl::desc("Location of the type typeDB directory"), cl::Hidden);

// FIXME 1) include bitcasts? 2) disabled by default in LLVM builds (use LLVM_ENABLE_STATS when building)
// STATISTIC(NumInstrumentedMallocs, "Number of instrumented mallocs");
// STATISTIC(NumInstrumentedFrees, "Number of instrumented frees");
STATISTIC(NumFoundMallocs, "Number of detected mallocs");
STATISTIC(NumFoundFrees, "Number of detected frees");
STATISTIC(NumFoundAlloca, "Number of detected (stack) allocas");

namespace tu = util::type;

namespace must {
namespace pass {

// Used by LLVM pass manager to identify passes in memory
char MustSupportPass::ID = 0;

// std::unique_ptr<TypeMapping> MustSupportPass::typeMapping = std::make_unique<SimpleTypeMapping>();

MustSupportPass::MustSupportPass() : llvm::BasicBlockPass(ID) {
  if (ClConfigDir.empty()) {
    configFile = std::string("./") + configFileName;
  } else {
    configFile = ClConfigDir + "/" + configFileName;
  }
}

bool MustSupportPass::doInitialization(Module& m) {
  /**
   * Introduce the necessary instrumentation functions in the LLVM module.
   * functions:
   * void __must_support_alloc(void *ptr_base, int type_id, long int count, long int elem_size)
   * void __must_support_free(void *ptr)
   *
   * Also scan the LLVM module for type definitions and add them to our type list.
   */
  declareInstrumentationFunctions(m);

  propagateTypeInformation(m);

  return true;
}

bool MustSupportPass::runOnBasicBlock(BasicBlock& bb) {
  /*
   * + Find malloc functions
   * + Find free frunctions
   * + Generate calls to instrumentation functions
   */
  namespace util = typeart::util;

  auto& c = bb.getContext();
  DataLayout dl(bb.getModule());

  MemOpVisitor mOpsCollector;
  mOpsCollector.visit(bb);

  // instrument collected calls of bb:
  for (auto& malloc : mOpsCollector.listMalloc) {
    ++NumFoundMallocs;

    auto mallocInst = malloc.call;
    const auto& bitcasts = malloc.bitcasts;

    BitCastInst* primaryBitcast = nullptr;
    for (auto bitcastInst : bitcasts) {
      if (!tu::isVoidPtr(bitcastInst->getDestTy())) {  // TODO: Any other types that should be ignored?
        // First non-void bitcast determines the type
        primaryBitcast = bitcastInst;
        break;
      }
    }

    // Number of bytes allocated
    auto mallocArg = mallocInst->getOperand(0);
    int typeId = typeManager.getOrRegisterType(mallocInst->getType()->getPointerElementType(),
                                               dl);  // retrieveTypeID(tu::getVoidType(c));
    // Number of bytes per element, 1 for void*
    unsigned typeSize = tu::getTypeSizeInBytes(mallocInst->getType()->getPointerElementType(), dl);
    auto insertBefore = mallocInst->getNextNode();

    // Use the first cast as the determining type (if there is any)
    if (primaryBitcast) {
      // primaryBitcast->dump();

      auto dstPtrType = primaryBitcast->getDestTy()->getPointerElementType();
      typeSize = tu::getTypeSizeInBytes(dstPtrType, dl);
      typeId = typeManager.getOrRegisterType(dstPtrType, dl);  //(unsigned)dstPtrType->getTypeID();
      insertBefore = primaryBitcast->getNextNode();

      // Handle additional bitcasts that occur after the first one
      std::for_each(std::next(bitcasts.begin()), bitcasts.end(), [&](auto bitcastInst) {
        if (!tu::isVoidPtr(bitcastInst->getDestTy()) && primaryBitcast->getDestTy() != bitcastInst->getDestTy()) {
          // Second non-void* bitcast detected - semantics unclear
          LOG_WARNING("Encountered ambiguous pointer type in allocation: " << util::dump(*mallocInst));
          LOG_WARNING("  Primary cast: " << util::dump(*primaryBitcast));
          LOG_WARNING("  Secondary cast: " << util::dump(*bitcastInst));
        }
      });
    }

    IRBuilder<> IRB(insertBefore);
    auto typeIdConst = ConstantInt::get(tu::getInt32Type(c), typeId);
    auto typeSizeConst = ConstantInt::get(tu::getInt64Type(c), typeSize);
    // Compute element count: count = numBytes / typeSize
    auto elementCount = IRB.CreateUDiv(mallocArg, typeSizeConst);
    IRB.CreateCall(typeart_alloc.f, ArrayRef<Value*>{mallocInst, typeIdConst, elementCount, typeSizeConst});
  }

  for (auto& free : mOpsCollector.listFree) {
    ++NumFoundFrees;
    // Pointer address:
    auto freeArg = free->getOperand(0);
    IRBuilder<> IRB(free->getNextNode());
    IRB.CreateCall(typeart_free.f, ArrayRef<Value*>{freeArg});
  }

#define INSTRUMENT_STACK_ALLOCS 0
#if INSTRUMENT_STACK_ALLOCS
  for (auto& alloca : mOpsCollector.listAlloca) {
    if (alloca->getAllocatedType()->isArrayTy()) {
      ++NumFoundAlloca;
      unsigned typeSize = tu::getTypeSizeForArrayAlloc(alloca, dl);
      auto insertBefore = alloca->getNextNode();

      auto elementType = alloca->getAllocatedType()->getArrayElementType();

      int typeId = typeManager.getOrRegisterType(elementType, dl);
      auto arraySize = alloca->getAllocatedType()->getArrayNumElements();

      auto typeIdConst = ConstantInt::get(tu::getInt32Type(c), typeId);
      auto typeSizeConst = ConstantInt::get(tu::getInt64Type(c), typeSize);
      auto numElementsConst = ConstantInt::get(tu::getInt64Type(c), arraySize);

      // Cast array to void*
      auto arrayPtr = CastInst::CreateBitOrPointerCast(alloca, tu::getVoidPtrType(c), "", insertBefore);

      // Call runtime lib
      std::vector<Value*> mustAllocaArgs{arrayPtr, typeIdConst, numElementsConst, typeSizeConst};
      CallInst::Create(mustAllocFn, mustAllocaArgs, "", arrayPtr->getNextNode());
    }
  }
#endif

  return false;
}

bool MustSupportPass::doFinalization(Module& m) {
  /*
   * Persist the accumulated type definition information for this module.
   */
  LOG_DEBUG("Writing type config file...");

  if (typeManager.store(configFile)) {
    LOG_DEBUG("Success!");
  } else {
    LOG_ERROR("Failed writing type config to " << configFile);
  }
  if (ClMustStats) {
    printStats(llvm::errs());
  }
  return false;
}

void MustSupportPass::declareInstrumentationFunctions(Module& m) {
  const auto make_function = [&m](auto& f_struct, auto f_type) {
    f_struct.f = m.getOrInsertFunction(f_struct.name, f_type);
    if (auto f = dyn_cast<Function>(f_struct.f)) {
      f->setLinkage(GlobalValue::ExternalLinkage);
    }
  };

  auto& c = m.getContext();
  Type* alloc_arg_types[] = {tu::getVoidPtrType(c), tu::getInt32Type(c), tu::getInt64Type(c), tu::getInt64Type(c)};
  Type* free_arg_types[] = {tu::getVoidPtrType(c)};

  make_function(typeart_alloc, FunctionType::get(Type::getVoidTy(c), alloc_arg_types, false));
  make_function(typeart_free, FunctionType::get(Type::getVoidTy(c), free_arg_types, false));
}

void MustSupportPass::propagateTypeInformation(Module& m) {
  /* Read already acquired information from temporary storage */
  /*
   * Scan module for type definitions and add to the type information map
   * Type information needed:
   *  + Name
   *  + Data member
   *  + Extent
   *  + Our id
   */
  if (typeManager.load(configFile)) {
    LOG_DEBUG("Existing type configuration successfully loaded");
  } else {
    LOG_DEBUG("No previous type configuration found");
  }
}

void MustSupportPass::printStats(llvm::raw_ostream& out) {
  const unsigned max_string{12u};
  const unsigned max_val{5u};
  std::string line(22, '-');
  line += "\n";
  const auto make_format = [&](const char* desc, const auto val) {
    return format("%-*s: %*u\n", max_string, desc, max_val, val);
  };

  out << line;
  out << "   MustSupportPass\n";
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
}  // namespace must

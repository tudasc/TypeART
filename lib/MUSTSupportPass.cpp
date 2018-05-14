#include "MUSTSupportPass.h"
#include "MemOpVisitor.h"
#include "support/Logger.h"
#include "support/TypeUtil.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Format.h"

#include <ConfigIO.h>
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

bool MustSupportPass::doInitialization(Function& f) {
  // TODO Do we actually need a per-function initialization?
  return false;
}

bool MustSupportPass::runOnBasicBlock(BasicBlock& bb) {
  /*
   * + Find malloc functions
   * + Find free frunctions
   * + Generate calls to instrumentation functions
   */

  auto& c = bb.getContext();
  DataLayout dl(bb.getModule());

  MemOpVisitor mOpsCollector;
  mOpsCollector.visit(bb);

  auto mustAllocFn = bb.getModule()->getFunction(allocInstrumentation);
  assert(mustAllocFn && "alloc instrumentation function not found");
  auto mustDeallocFn = bb.getModule()->getFunction(freeInstrumentation);
  assert(mustDeallocFn && "free instrumentation function not found");

  // instrument collected calls of bb:
  for (auto& malloc : mOpsCollector.listMalloc) {
    ++NumFoundMallocs;

    auto mallocInst = malloc.call;

    BitCastInst* primaryBitcast = nullptr;
    auto bitcastIt = malloc.bitcasts.begin();
    for (; bitcastIt != malloc.bitcasts.end(); bitcastIt++) {
      auto bitcastInst = *bitcastIt;
      // auto dstPtrType = bitcastInst->getDestTy()->getPointerElementType();
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
      bitcastIt++;
      for (; bitcastIt != malloc.bitcasts.end(); bitcastIt++) {
        auto bitcastInst = *bitcastIt;
        // Casts to void* can be ignored
        if (!tu::isVoidPtr(bitcastInst->getDestTy()) && primaryBitcast->getDestTy() != bitcastInst->getDestTy()) {
          // Second non-void* bitcast detected - semantics unclear
          LOG_WARNING("Encountered ambiguous pointer type in allocation:");  // TODO: Better warning message
          mallocInst->dump();
          LOG_WARNING("Primary cast:");
          primaryBitcast->dump();
          LOG_WARNING("Secondary cast:");
          bitcastInst->dump();
        }
      }
    }

    // mallocInst->dump();

    auto typeIdConst = ConstantInt::get(tu::getInt32Type(c), typeId);
    auto typeSizeConst = ConstantInt::get(tu::getInt64Type(c), typeSize);
    // Compute element count: count = numBytes / typeSize
    auto elementCount = BinaryOperator::CreateUDiv(mallocArg, typeSizeConst, "", insertBefore);

    // Call runtime lib
    std::vector<Value*> mustAllocArgs{mallocInst, typeIdConst, elementCount, typeSizeConst};
    CallInst::Create(mustAllocFn, mustAllocArgs, "", elementCount->getNextNode());
  }

  for (auto& free : mOpsCollector.listFree) {
    ++NumFoundFrees;
    // Pointer address
    auto freeArg = free->getOperand(0);
    auto insertBefore = free->getNextNode();
    // Call runtime lib
    std::vector<Value*> mustFreeArgs{freeArg};
    CallInst::Create(mustDeallocFn, mustFreeArgs, "", insertBefore);
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
  auto file = "/tmp/musttypes";
  if (typeManager.store(file)) {
    LOG_DEBUG("Success!");
  } else {
    LOG_ERROR("Failed writing type config to " << file);
  }

  if (ClMustStats) {
    printStats(llvm::errs());
  }
  return false;
}

void MustSupportPass::setFunctionLinkageExternal(llvm::Constant* c) {
  if (auto f = dyn_cast<Function>(c)) {
    assert(f != nullptr && "The function pointer is not null");
    f->setLinkage(GlobalValue::ExternalLinkage);
  }
}

void MustSupportPass::declareInstrumentationFunctions(Module& m) {
  auto& c = m.getContext();
  auto allocFunc = m.getOrInsertFunction(allocInstrumentation, tu::getVoidType(c), tu::getVoidPtrType(c),
                                         tu::getInt32Type(c), tu::getInt64Type(c), tu::getInt64Type(c), nullptr);
  setFunctionLinkageExternal(allocFunc);

  auto freeFunc = m.getOrInsertFunction(freeInstrumentation, tu::getVoidType(c), tu::getVoidPtrType(c), nullptr);
  setFunctionLinkageExternal(freeFunc);
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
  if (typeManager.load("/tmp/musttypes")) {
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

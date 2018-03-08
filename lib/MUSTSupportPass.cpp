#include "MUSTSupportPass.h"
#include "MemOpVisitor.h"
#include "support/Logger.h"
#include "support/TypeUtil.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Format.h"

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

std::unique_ptr<TypeMapping> MustSupportPass::typeMapping = std::make_unique<SimpleTypeMapping>();

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

    // FIXME: Problem when the result of malloc is not casted immediately (see 10_malloc_multiple_casts.c).

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
    // Number of bytes per element, 1 for void*
    unsigned typeSize = 1;
    int typeId = typeMapping->getTypeId(tu::getVoidType(c));  // FIXME: use void type id as default
    auto insertBefore = mallocInst->getNextNode();

    if (primaryBitcast) {
      auto dstPtrType = primaryBitcast->getDestTy()->getPointerElementType();
      typeSize = tu::getTypeSizeInBytes(dstPtrType, dl);
      // TODO: Implement sensible type mapping
      typeId = typeMapping->getTypeId(dstPtrType);  //(unsigned)dstPtrType->getTypeID();
      insertBefore = primaryBitcast->getNextNode();

      // Handle additional bitcasts that occur after the first one
      bitcastIt++;
      for (; bitcastIt != malloc.bitcasts.end(); bitcastIt++) {
        auto bitcastInst = *bitcastIt;
        // Casts to void* can be ignored
        if (!tu::isVoidPtr(bitcastInst->getDestTy())) {
          // Second non-void* bitcast detected - semantics unclear
          LOG_WARNING("Encountered ambiguous pointer type");  // TODO: Better warning message
        }
      }
    }

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
  }
  for (auto& alloca : mOpsCollector.listAlloca) {
    ++NumFoundAlloca;
  }

  return false;
}

bool MustSupportPass::doFinalization(Module& m) {
  /*
   * Persist the accumulated type definition information for this module.
   */
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

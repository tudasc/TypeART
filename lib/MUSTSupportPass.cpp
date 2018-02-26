#include "MUSTSupportPass.h"
#include "TypeUtil.h"
#include <llvm/IR/Constants.h>

#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace {
static llvm::RegisterPass<must::pass::MustSupportPass> msp("must", "MUST type information", false, false);
}  // namespace

namespace tu = util::type;

namespace must {
namespace pass {

// Used by LLVM pass manager to identify passes in memory
char MustSupportPass::ID = 0;

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

  for (Instruction& inst : bb.getInstList()) {
    switch (inst.getOpcode()) {
      case Instruction::BitCast: {
        auto bitcastInst = dyn_cast<BitCastInst>(&inst);
        auto srcValue = bitcastInst->getOperand(0);
        if (auto callInst = dyn_cast<CallInst>(&inst)) {
          auto callee = callInst->getCalledFunction();
          auto calleeName = callee->getName();
          if (isAllocateFunction(calleeName)) {
            // Instrument allocation
            // 1. Find out size
            // 2. Find out type
            auto mallocArg = callee->getOperand(0);  // Number of bytes
            // if (auto constIntArg = dyn_cast<ConstantInt>(mallocArg)) {
            // constIntArg->
            //}

            auto dstPtrType = bitcastInst->getDestTy()->getPointerElementType();
            auto typeSize = tu::getTypeSizeInBytes(dstPtrType, dl);

            // TODO: Type IDs for structs etc.
            auto typeId = dstPtrType->getTypeID();

            auto mustAllocFn = bb.getModule()->getFunction(allocInstrumentation);
            // TODO: Ensure function exists

            auto typeIdConst = ConstantInt::get(tu::getInt32Type(c), (unsigned)typeId);
            auto typeSizeConst = ConstantInt::get(tu::getInt64Type(c), typeSize);

            // count = numBytes / typeSize
            auto elementCount = BinaryOperator::CreateUDiv(mallocArg, typeSizeConst, "", bitcastInst->getNextNode());

            std::vector<Value*> mustAllocArgs{callee, typeIdConst, elementCount, typeSizeConst};
            CallInst::Create(mustAllocFn, mustAllocArgs, "", elementCount->getNextNode());
          }
        }
        break;
      }
      case Instruction::Call: {
        auto callInst = dyn_cast<CallInst>(&inst);
        auto callee = callInst->getCalledFunction();
        auto calleeName = callee->getName();
        if (isAllocateFunction(calleeName)) {
          // Instrument allocation

        } else if (isDeallocateFunction(calleeName)) {
          // Instrument deallocation
        }
        break;
      }
      default:
        break;
    }
  }
  return false;
}

bool MustSupportPass::doFinalization(Module& m) {
  /*
   * Persist the accumulated type definition information for this module.
   */
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
  auto allocFunc = m.getOrInsertFunction(allocInstrumentation, tu::getVoidPtrType(c), tu::getInt32Type(c),
                                         tu::getInt64Type(c), tu::getInt64Type(c), nullptr);
  setFunctionLinkageExternal(allocFunc);
  this->mustSupportAllocFn = allocFunc;

  auto freeFunc = m.getOrInsertFunction(freeInstrumentation, tu::getVoidPtrType(c), nullptr);
  setFunctionLinkageExternal(freeFunc);
  this->mustSupportFreeFn = freeFunc;
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

}  // namespace pass
}  // namespace must

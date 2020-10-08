//
// Created by ahueck on 08.10.20.
//

#include "../TypeManager.h"
#include "ArgCollector.h"
#include "InstrumentationHelper.h"
#include "support/Logger.h"
#include "support/TypeUtil.h"
#include "support/Util.h"

#include "llvm/IR/Instructions.h"

namespace tu = typeart::util::type;
using namespace llvm;
namespace typeart {

MemOpArgCollector::MemOpArgCollector(TypeManager& tm, InstrumentationHelper& instr)
    : ArgumentCollector(), tm(tm), instr(instr) {
}
HeapArgList MemOpArgCollector::collectHeap(const llvm::SmallVectorImpl<MallocData>& mallocs) {
  HeapArgList list;
  list.reserve(mallocs.size());
  const llvm::DataLayout& dl = instr.getModule()->getDataLayout();
  for (const MallocData& mdata : mallocs) {
    const auto malloc_call      = mdata.call;
    BitCastInst* primaryBitcast = mdata.primary;
    const bool is_invoke        = mdata.is_invoke;
    auto kind                   = mdata.kind;

    // Number of bytes allocated
    auto mallocArg = malloc_call->getOperand(0);
    int typeId     = tm.getOrRegisterType(malloc_call->getType()->getPointerElementType(),
                                      dl);  // retrieveTypeID(tu::getVoidType(c));
    if (typeId == TA_UNKNOWN_TYPE) {
      LOG_ERROR("Unknown allocated type. Not instrumenting. " << util::dump(*malloc_call));
      // TODO notify caller that we skipped: via lambda callback function
      continue;
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

      typeId = tm.getOrRegisterType(dstPtrType, dl);
      if (typeId == TA_UNKNOWN_TYPE) {
        LOG_ERROR("Target type of casted allocation is unknown. Not instrumenting. " << util::dump(*malloc_call));
        LOG_ERROR("Cast: " << util::dump(*primaryBitcast));
        LOG_ERROR("Target type: " << util::dump(*dstPtrType));
        // TODO notify caller that we skipped: via lambda callback function
        continue;
      }
    } else {
      LOG_ERROR("Primary bitcast is null. malloc: " << util::dump(*malloc_call))
    }

    auto* typeIdConst   = instr.getConstantFor(IType::type_id, typeId);
    auto* typeSizeConst = instr.getConstantFor(IType::extent, typeSize);
    // Compute element count: count = numBytes / typeSize
    Value* elementCount = nullptr;
    Value* byte_count{nullptr};
    switch (kind) {
      case MemOpKind::MALLOC:
        // elementCount = IRB.CreateUDiv(mallocArg, typeSizeConst);
        byte_count = mallocArg;
      case MemOpKind::CALLOC:
        // elementCount = malloc_call->getOperand(0);  // get the element count in calloc call
        byte_count = malloc_call->getOperand(0);
      case MemOpKind::REALLOC:
        // auto mArg    = malloc_call->getOperand(1);
        // elementCount = IRB.CreateUDiv(mArg, typeSizeConst);

        byte_count = malloc_call->getOperand(1);

        // IRBuilder<> FreeB(malloc_call);
        // auto addrOp = malloc_call->getOperand(0);
        // FreeB.CreateCall(typeart_free.f, ArrayRef<Value*>{addrOp});
      default:
        LOG_ERROR("Unknown malloc kind. Not instrumenting. " << util::dump(*malloc_call));
        // TODO see above continues
        continue;
    }

    list.emplace_back(HeapArgList::value_type{mdata, {malloc_call, typeIdConst, typeSizeConst, byte_count}});
    // IRB.CreateCall(typeart_alloc.f, ArrayRef<Value*>{malloc_call, typeIdConst, elementCount});
  }

  return list;
}
FreeArgList MemOpArgCollector::collectFree(const llvm::SmallVectorImpl<FreeData>& frees) {
  FreeArgList list;
  list.reserve(frees.size());
  for (const FreeData& fdata : frees) {
    auto free_call = fdata.call;
    auto freeArg   = free_call->getOperand(0);

    list.emplace_back(FreeArgList::value_type{fdata, {freeArg}});
  }

  return list;
}
StackArgList MemOpArgCollector::collectStack(const llvm::SmallVectorImpl<AllocaData>& allocs) {
  using namespace llvm;
  StackArgList list;
  list.reserve(allocs.size());
  const llvm::DataLayout& dl = instr.getModule()->getDataLayout();

  for (const AllocaData& adata : allocs) {
    auto alloca           = adata.alloca;
    Type* elementType     = alloca->getAllocatedType();
    Value* numElementsVal = nullptr;
    // The length can be specified statically through the array type or as a separate argument.
    // Both cases are handled here.
    if (adata.is_vla) {
      numElementsVal = alloca->getArraySize();
      // This should not happen in generated IR code
      assert(!elementType->isArrayTy() && "VLAs of array types are currently not supported.");
    } else {
      size_t arraySize = adata.array_size;
      if (elementType->isArrayTy()) {
        arraySize   = arraySize * tu::getArrayLengthFlattened(elementType);
        elementType = tu::getArrayElementType(elementType);
      }
      numElementsVal = instr.getConstantFor(IType::extent, arraySize);
    }

    // unsigned typeSize = tu::getTypeSizeInBytes(elementType, dl);
    int typeId = tm.getOrRegisterType(elementType, dl);

    if (typeId == TA_UNKNOWN_TYPE) {
      LOG_ERROR("Type is not supported: " << util::dump(*elementType));
    }

    auto* typeIdConst = instr.getConstantFor(IType::type_id, typeId);

    list.emplace_back(StackArgList::value_type{adata, {typeIdConst, numElementsVal}});
  }

  return list;
}
GlobalArgList MemOpArgCollector::collectGlobal(const llvm::SmallVectorImpl<GlobalData>& globals) {
  GlobalArgList list;
  list.reserve(globals.size());
  for (const GlobalData& gdata : globals) {
  }

  return list;
}
}  // namespace typeart
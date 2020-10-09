//
// Created by ahueck on 09.10.20.
//

#include "MemOpInstrumentation.h"

#include "../TypeManager.h"
#include "InstrumentationHelper.h"
#include "support/Logger.h"
#include "support/TypeUtil.h"
#include "support/Util.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

namespace typeart {
MemOpInstrumentation::MemOpInstrumentation(InstrumentationHelper& instr) : MemoryInstrument(), instr(instr) {
}

size_t MemOpInstrumentation::instrumentHeap(const HeapArgList& heap) {
  for (auto& heap_elem : heap) {
    const auto& malloc       = heap_elem.mem_data;
    auto kind                = malloc.kind;
    auto& args               = heap_elem.args;
    Instruction* malloc_call = args.get_as<Instruction>("pointer");

    Instruction* insertBefore = malloc_call->getNextNode();
    if (malloc.is_invoke) {
      const InvokeInst* inv = dyn_cast<InvokeInst>(malloc_call);
      insertBefore          = &(*inv->getNormalDest()->getFirstInsertionPt());
    }

    IRBuilder<> IRB(insertBefore);

    auto typeIdConst   = args.get_value("type_id");
    auto typeSizeConst = args.get_value("type_size");
    Value* elementCount{nullptr};

    switch (kind) {
      case MemOpKind::MALLOC: {
        auto bytes   = args.get_value("byte_count");  // can be null (for calloc, realloc)
        elementCount = IRB.CreateUDiv(bytes, typeSizeConst);
        break;
      }
      case MemOpKind::CALLOC: {
        elementCount = args.get_value("element_count");
        break;
      }
      case MemOpKind::REALLOC: {
        auto mArg   = args.get_value("element_count");
        auto addrOp = args.get_value("realloc_ptr");

        elementCount = IRB.CreateUDiv(mArg, typeSizeConst);
        IRBuilder<> FreeB(malloc_call);
        // FreeB.CreateCall(typeart_free.f, ArrayRef<Value*>{addrOp});
        break;
      }
      default:
        LOG_ERROR("Unknown malloc kind. Not instrumenting. " << util::dump(*malloc_call));
        continue;
    }

    // IRB.CreateCall(typeart_alloc.f, ArrayRef<Value*>{malloc_call, typeIdConst, elementCount});
  }
  return 0;
}
size_t MemOpInstrumentation::instrumentFree(const FreeArgList& frees) {
  return 0;
}
size_t MemOpInstrumentation::instrumentStack(const StackArgList& frees) {
  return 0;
}
size_t MemOpInstrumentation::instrumentGlobal(const GlobalArgList& globals) {
  return 0;
}
}  // namespace typeart
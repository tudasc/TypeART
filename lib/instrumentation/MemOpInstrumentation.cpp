//
// Created by ahueck on 09.10.20.
//

#include "MemOpInstrumentation.h"

#include "../TypeManager.h"
#include "InstrumentationHelper.h"
#include "TransformUtil.h"
#include "TypeARTFunctions.h"
#include "support/Logger.h"
#include "support/Util.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/CtorUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

namespace typeart {
MemOpInstrumentation::MemOpInstrumentation(TAFunctionQuery& fquery, InstrumentationHelper& instr)
    : MemoryInstrument(), fquery(fquery), instr(instr) {
}

size_t MemOpInstrumentation::instrumentHeap(const HeapArgList& heap) {
  unsigned counter{0};
  for (auto& [malloc, args] : heap) {
    auto kind                = malloc.kind;
    Instruction* malloc_call = args.get_as<Instruction>(ArgMap::ID::pointer);

    Instruction* insertBefore = malloc_call->getNextNode();
    if (malloc.is_invoke) {
      const InvokeInst* inv = dyn_cast<InvokeInst>(malloc_call);
      insertBefore          = &(*inv->getNormalDest()->getFirstInsertionPt());
    }

    IRBuilder<> IRB(insertBefore);

    auto typeIdConst   = args.get_value(ArgMap::ID::type_id);
    auto typeSizeConst = args.get_value(ArgMap::ID::type_size);
    Value* elementCount{nullptr};

    switch (kind) {
      case MemOpKind::MALLOC: {
        auto bytes   = args.get_value(ArgMap::ID::byte_count);  // can be null (for calloc, realloc)
        elementCount = IRB.CreateUDiv(bytes, typeSizeConst);
        break;
      }
      case MemOpKind::CALLOC: {
        elementCount = args.get_value(ArgMap::ID::element_count);
        break;
      }
      case MemOpKind::REALLOC: {
        auto mArg   = args.get_value(ArgMap::ID::byte_count);
        auto addrOp = args.get_value(ArgMap::ID::realloc_ptr);

        elementCount = IRB.CreateUDiv(mArg, typeSizeConst);
        IRBuilder<> FreeB(malloc_call);
        FreeB.CreateCall(fquery.getFunctionFor(IFunc::free), ArrayRef<Value*>{addrOp});
        break;
      }
      default:
        LOG_ERROR("Unknown malloc kind. Not instrumenting. " << util::dump(*malloc_call));
        continue;
    }

    IRB.CreateCall(fquery.getFunctionFor(IFunc::heap), ArrayRef<Value*>{malloc_call, typeIdConst, elementCount});
    ++counter;
  }

  return counter;
}
size_t MemOpInstrumentation::instrumentFree(const FreeArgList& frees) {
  unsigned counter{0};
  for (auto& [fdata, args] : frees) {
    auto free_call       = fdata.call;
    const bool is_invoke = fdata.is_invoke;

    Instruction* insertBefore = free_call->getNextNode();
    if (is_invoke) {
      InvokeInst* inv = dyn_cast<InvokeInst>(free_call);
      insertBefore    = &(*inv->getNormalDest()->getFirstInsertionPt());
    }

    auto free_arg = args.get_value(ArgMap::ID::pointer);

    IRBuilder<> IRB(insertBefore);
    IRB.CreateCall(fquery.getFunctionFor(IFunc::free), ArrayRef<Value*>{free_arg});
    ++counter;
  }

  return counter;
}
size_t MemOpInstrumentation::instrumentStack(const StackArgList& stack) {
  using namespace transform;
  unsigned counter{0};
  StackCounter::StackOpCounter allocCounts;
  Function* f{nullptr};
  for (auto& [sdata, args] : stack) {
    // auto alloca = sdata.alloca;
    Instruction* alloca = args.get_as<Instruction>(ArgMap::ID::pointer);

    IRBuilder<> IRB(alloca->getNextNode());

    auto typeIdConst    = args.get_value(ArgMap::ID::type_id);
    auto numElementsVal = args.get_value(ArgMap::ID::element_count);
    auto arrayPtr       = IRB.CreateBitOrPointerCast(alloca, instr.getTypeFor(IType::ptr));

    IRB.CreateCall(fquery.getFunctionFor(IFunc::stack), ArrayRef<Value*>{arrayPtr, typeIdConst, numElementsVal});
    ++counter;

    auto bb = alloca->getParent();
    allocCounts[bb]++;
    if (!f) {
      f = bb->getParent();
    }
  }

  if (f) {
    StackCounter scount(f, instr, fquery);
    scount.addStackHandling(allocCounts);
  }

  return counter;
}
size_t MemOpInstrumentation::instrumentGlobal(const GlobalArgList& globals) {
  unsigned counter{0};

  const auto instrumentGlobalsInCtor = [&](auto& IRB) {
    for (auto& [gdata, args] : globals) {
      // Instruction* global = args.get_as<llvm::Instruction>("pointer");
      auto global         = gdata.global;
      auto typeIdConst    = args.get_value(ArgMap::ID::type_id);
      auto numElementsVal = args.get_value(ArgMap::ID::element_count);
      auto globalPtr      = IRB.CreateBitOrPointerCast(global, instr.getTypeFor(IType::ptr));
      IRB.CreateCall(fquery.getFunctionFor(IFunc::global), ArrayRef<Value*>{globalPtr, typeIdConst, numElementsVal});
      ++counter;
    }
  };

  const auto makeCtorFuncBody = [&]() -> IRBuilder<> {
    auto m                = instr.getModule();
    auto& c               = m->getContext();
    auto ctorFunctionName = "__typeart_init_module_" + m->getSourceFileName();  // needed -- will not work with piping?

    FunctionType* ctorType = FunctionType::get(llvm::Type::getVoidTy(c), false);
    Function* ctorFunction = Function::Create(ctorType, Function::PrivateLinkage, ctorFunctionName, m);

    BasicBlock* entry = BasicBlock::Create(c, "entry", ctorFunction);

    llvm::appendToGlobalCtors(*m, ctorFunction, 0, nullptr);

    IRBuilder<> IRB(entry);
    return IRB;
  };

  auto IRB = makeCtorFuncBody();
  instrumentGlobalsInCtor(IRB);
  IRB.CreateRetVoid();

  return counter;
}
}  // namespace typeart
// TypeART library
//
// Copyright (c) 2017-2022 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#include "MemOpInstrumentation.h"

#include "Instrumentation.h"
#include "InstrumentationHelper.h"
#include "TransformUtil.h"
#include "TypeARTFunctions.h"
#include "analysis/MemOpData.h"
#include "support/Logger.h"
#include "support/OmpUtil.h"
#include "support/Util.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <string>

namespace llvm {
class Value;
}  // namespace llvm

using namespace llvm;

namespace typeart {

MemOpInstrumentation::MemOpInstrumentation(TAFunctionQuery& fquery, InstrumentationHelper& instr,
                                           bool lifetime_instrument)
    : MemoryInstrument(), fquery(&fquery), instr_helper(&instr), instrument_lifetime(lifetime_instrument) {
}

InstrCount MemOpInstrumentation::instrumentHeap(const HeapArgList& heap) {
  InstrCount counter{0};
  for (const auto& [malloc, args] : heap) {
    auto kind                = malloc.kind;
    Instruction* malloc_call = args.get_as<Instruction>(ArgMap::ID::pointer);

    Instruction* insertBefore = malloc_call->getNextNode();
    if (malloc.array_cookie.hasValue()) {
      insertBefore = malloc.array_cookie.getValue().array_ptr_gep->getNextNode();
    } else {
      if (malloc.is_invoke) {
        const InvokeInst* inv = dyn_cast<InvokeInst>(malloc_call);
        insertBefore          = &(*inv->getNormalDest()->getFirstInsertionPt());
      }
    }

    IRBuilder<> IRB(insertBefore);

    auto typeIdConst   = args.get_value(ArgMap::ID::type_id);
    auto typeSizeConst = args.get_value(ArgMap::ID::type_size);

    bool single_byte_type{false};
    if (auto* const_int = llvm::dyn_cast<ConstantInt>(typeSizeConst)) {
      single_byte_type = const_int->equalsInt(1);
    }

    Value* elementCount{nullptr};

    auto parent_f  = malloc.call->getFunction();
    const bool omp = util::omp::isOmpContext(parent_f);

    switch (kind) {
      case MemOpKind::AlignedAllocLike:
        [[fallthrough]];
      case MemOpKind::NewLike:
        [[fallthrough]];
      case MemOpKind::MallocLike: {
        elementCount = args.lookup(ArgMap::ID::element_count);
        if (elementCount == nullptr) {
          auto bytes   = args.get_value(ArgMap::ID::byte_count);  // can be null (for calloc, realloc)
          elementCount = single_byte_type ? bytes : IRB.CreateUDiv(bytes, typeSizeConst);
        }
        break;
      }
      case MemOpKind::CallocLike: {
        if (malloc.primary == nullptr) {
          auto elems     = args.get_value(ArgMap::ID::element_count);
          auto type_size = args.get_value(ArgMap::ID::type_size);
          elementCount   = IRB.CreateMul(elems, type_size);
        } else {
          elementCount = args.get_value(ArgMap::ID::element_count);
        }
        break;
      }
      case MemOpKind::ReallocLike: {
        auto mArg   = args.get_value(ArgMap::ID::byte_count);
        auto addrOp = args.get_value(ArgMap::ID::realloc_ptr);

        elementCount = single_byte_type ? mArg : IRB.CreateUDiv(mArg, typeSizeConst);
        IRBuilder<> FreeB(malloc_call);
        const auto callback_id = omp ? IFunc::free_omp : IFunc::free;
        FreeB.CreateCall(fquery->getFunctionFor(callback_id), ArrayRef<Value*>{addrOp});
        break;
      }
      default:
        LOG_ERROR("Unknown malloc kind. Not instrumenting. " << util::dump(*malloc_call));
        continue;
    }

    const auto callback_id = omp ? IFunc::heap_omp : IFunc::heap;
    IRB.CreateCall(fquery->getFunctionFor(callback_id), ArrayRef<Value*>{malloc_call, typeIdConst, elementCount});
    ++counter;
  }

  return counter;
}

InstrCount MemOpInstrumentation::instrumentFree(const FreeArgList& frees) {
  InstrCount counter{0};
  for (const auto& [fdata, args] : frees) {
    auto free_call       = fdata.call;
    const bool is_invoke = fdata.is_invoke;

    Instruction* insertBefore = free_call->getNextNode();
    if (is_invoke) {
      auto* inv    = dyn_cast<InvokeInst>(free_call);
      insertBefore = &(*inv->getNormalDest()->getFirstInsertionPt());
    }

    Value* free_arg{nullptr};
    switch (fdata.kind) {
      case MemOpKind::DeleteLike:
        [[fallthrough]];
      case MemOpKind::FreeLike:
        free_arg = args.get_value(ArgMap::ID::pointer);
        break;
      default:
        LOG_ERROR("Unknown free kind. Not instrumenting. " << util::dump(*free_call));
        continue;
    }

    IRBuilder<> IRB(insertBefore);

    auto parent_f          = fdata.call->getFunction();
    const auto callback_id = util::omp::isOmpContext(parent_f) ? IFunc::free_omp : IFunc::free;

    IRB.CreateCall(fquery->getFunctionFor(callback_id), ArrayRef<Value*>{free_arg});
    ++counter;
  }

  return counter;
}

InstrCount MemOpInstrumentation::instrumentStack(const StackArgList& stack) {
  using namespace transform;
  InstrCount counter{0};
  StackCounter::StackOpCounter allocCounts;
  Function* function{nullptr};
  for (const auto& [sdata, args] : stack) {
    auto* alloca         = args.get_as<Instruction>(ArgMap::ID::pointer);
    auto* typeIdConst    = args.get_value(ArgMap::ID::type_id);
    auto* numElementsVal = args.get_value(ArgMap::ID::element_count);

    const auto instrument_stack = [&](IRBuilder<>& IRB, Value* data_ptr, Instruction* anchor) {
      const auto callback_id = util::omp::isOmpContext(anchor->getFunction()) ? IFunc::stack_omp : IFunc::stack;
      IRB.CreateCall(fquery->getFunctionFor(callback_id), ArrayRef<Value*>{data_ptr, typeIdConst, numElementsVal});
      ++counter;

      auto* bblock = anchor->getParent();
      allocCounts[bblock]++;
      if (function == nullptr) {
        function = bblock->getParent();
      }
    };

    const auto& lifetime_starts = sdata.lifetime_start;
    if (lifetime_starts.empty() || !instrument_lifetime) {
      IRBuilder<> IRB(alloca->getNextNode());
      auto* data_ptr = IRB.CreateBitOrPointerCast(alloca, instr_helper->getTypeFor(IType::ptr));
      instrument_stack(IRB, data_ptr, alloca);
    } else {
      for (auto* lifetime_s : lifetime_starts) {
        IRBuilder<> IRB(lifetime_s->getNextNode());
        instrument_stack(IRB, lifetime_s->getOperand(1), lifetime_s->getNextNode());
      }
    }
  }

  if (function != nullptr) {
    StackCounter scount(function, instr_helper, fquery);
    scount.addStackHandling(allocCounts);
  }

  return counter;
}

InstrCount MemOpInstrumentation::instrumentGlobal(const GlobalArgList& globals) {
  InstrCount counter{0};

  const auto instrumentGlobalsInCtor = [&](auto& IRB) {
    for (const auto& [gdata, args] : globals) {
      // Instruction* global = args.get_as<llvm::Instruction>("pointer");
      auto global         = gdata.global;
      auto typeIdConst    = args.get_value(ArgMap::ID::type_id);
      auto numElementsVal = args.get_value(ArgMap::ID::element_count);
      auto globalPtr      = IRB.CreateBitOrPointerCast(global, instr_helper->getTypeFor(IType::ptr));
      IRB.CreateCall(fquery->getFunctionFor(IFunc::global), ArrayRef<Value*>{globalPtr, typeIdConst, numElementsVal});
      ++counter;
    }
  };

  const auto makeCtorFuncBody = [&]() -> BasicBlock* {
    auto m  = instr_helper->getModule();
    auto& c = m->getContext();
    auto ctorFunctionName =
        "__typeart_init_module_globals";  // + m->getSourceFileName();  // needed -- will not work with piping?

    FunctionType* ctorType = FunctionType::get(llvm::Type::getVoidTy(c), false);
    Function* ctorFunction = Function::Create(ctorType, Function::InternalLinkage, ctorFunctionName, m);

    BasicBlock* entry = BasicBlock::Create(c, "entry", ctorFunction);

    llvm::appendToGlobalCtors(*m, ctorFunction, 0, nullptr);

    return entry;
  };

  auto* entry = makeCtorFuncBody();
  IRBuilder<> IRB(entry);
  instrumentGlobalsInCtor(IRB);
  IRB.CreateRetVoid();

  return counter;
}

}  // namespace typeart
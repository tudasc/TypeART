// TypeART library
//
// Copyright (c) 2017-2021 TypeART Authors
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
#include "runtime/ParityConstant.h"
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

MemOpInstrumentation::MemOpInstrumentation(TAFunctionQuery& fquery, InstrumentationHelper& instr)
    : MemoryInstrument(), fquery(&fquery), instr_helper(&instr) {
}

InstrCount MemOpInstrumentation::instrumentHeap(const HeapArgList& heap) {
  InstrCount counter{0};
  for (const auto& [malloc, args] : heap) {
    auto kind                = malloc.kind;
    Instruction* malloc_call = args.get_as<Instruction>(ArgMap::ID::pointer);

    // Add space to prepend the pointer info
    if (auto call = llvm::dyn_cast<CallInst>(malloc_call)) {
      auto size_arg = call->getArgOperand(0);
      if (auto arg = llvm::dyn_cast<llvm::ConstantInt>(size_arg)) {
        call->setArgOperand(0, llvm::ConstantInt::get(arg->getType(), arg->getValue() + 32));
      } else {
        if (auto arg = llvm::dyn_cast<llvm::Instruction>(size_arg)) {
          IRBuilder<> arg_builder(malloc_call);
          call->setArgOperand(0, arg_builder.CreateAdd(arg, llvm::ConstantInt::get(arg->getType(), 32, true)));
        } else {
          llvm::errs() << "TODO unknown malloc arg type\n";
        }
      }
    }

    // Insert the PointerInfo struct
    auto& ctx = malloc_call->getContext();
    auto type = malloc_call->getModule()->getTypeByName("Typeart_PointerInfo");
    if (type == nullptr) {
      type = llvm::StructType::create({instr_helper->getTypeFor(IType::type_id),
                                       instr_helper->getTypeFor(IType::extent), llvm::Type::getInt64Ty(ctx)},
                                      "Typeart_PointerInfo");
    }
    auto ptr_type = llvm::PointerType::getUnqual(type);

    Instruction* insertBefore = malloc_call->getNextNode();
    Value* address            = malloc_call;
    if (malloc.array_cookie.hasValue()) {
      auto array_cookie = malloc.array_cookie.getValue();
      insertBefore      = array_cookie.array_ptr_gep->getNextNode();
      address           = array_cookie.array_ptr_gep;
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

        // TODO
        IRBuilder<> FreeB(malloc_call);
        const auto callback_id = omp ? IFunc::free_omp : IFunc::free;
        FreeB.CreateCall(fquery->getFunctionFor(callback_id), ArrayRef<Value*>{addrOp});
        break;
      }
      default:
        LOG_ERROR("Unknown malloc kind. Not instrumenting. " << util::dump(*malloc_call));
        continue;
    }

    // Write Fat Pointer data
    auto ptr_info = llvm::dyn_cast<BitCastInst>(IRB.CreateBitCast(address, ptr_type));
    auto type_id  = IRB.CreateStructGEP(ptr_info, 0);
    IRB.CreateStore(typeIdConst, type_id);
    auto count = IRB.CreateStructGEP(ptr_info, 1);
    IRB.CreateStore(elementCount, count);
    auto parity = IRB.CreateStructGEP(ptr_info, 2);
    IRB.CreateStore(llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx), parity_constant), parity);

    // Offset original pointer and replace uses
    auto user_data = llvm::dyn_cast<GetElementPtrInst>(
        IRB.CreateInBoundsGEP(address, llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx), 32)));
    address->replaceAllUsesWith(user_data);
    ptr_info->setOperand(0, address);
    user_data->setOperand(0, address);

    // TODO
    const auto callback_id = omp ? IFunc::heap_omp : IFunc::heap;
    IRB.CreateCall(fquery->getFunctionFor(callback_id), ArrayRef<Value*>{user_data, typeIdConst, elementCount});

    ++counter;
  }

  return counter;
}

InstrCount MemOpInstrumentation::instrumentFree(const FreeArgList& frees) {
  InstrCount counter{0};
  for (const auto& [fdata, args] : frees) {
    auto free_call       = fdata.call;
    auto& ctx            = free_call->getContext();
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
    // Reapply the Fat Pointer offset.
    IRBuilder<> free_builder(llvm::dyn_cast<Instruction>(free_arg)->getNextNode());
    auto original_ptr = llvm::dyn_cast<GetElementPtrInst>(free_builder.CreateInBoundsGEP(
        free_arg, llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx), llvm::APInt(64, 0, true) - 32)));
    free_arg->replaceAllUsesWith(original_ptr);
    original_ptr->setOperand(0, free_arg);

    // TODO
    auto parent_f          = fdata.call->getFunction();
    const auto callback_id = util::omp::isOmpContext(parent_f) ? IFunc::free_omp : IFunc::free;
    IRB.CreateCall(fquery->getFunctionFor(callback_id), ArrayRef<Value*>{free_arg});

    ++counter;
  }

  return counter;
}

std::string typeNameFor(llvm::Type* ty) {
  if (ty->isPointerTy()) {
    return typeNameFor(llvm::dyn_cast<PointerType>(ty)->getElementType()) + "*";
  } else if (ty->isStructTy()) {
    return ty->getStructName();
  } else if (ty->isIntegerTy()) {
    auto int_ty = llvm::dyn_cast<IntegerType>(ty);
    return std::string{"i"} + std::to_string(int_ty->getBitWidth());
  } else if (ty->isArrayTy()) {
    auto arr_ty = llvm::dyn_cast<ArrayType>(ty);
    return std::string{"["} + std::to_string(arr_ty->getNumElements()) + " x " + typeNameFor(arr_ty->getElementType()) +
           "]";
  } else if (ty->isFunctionTy()) {
    auto fn_ty = llvm::dyn_cast<FunctionType>(ty);
    auto param = std::string("");
    for (auto& arg : fn_ty->params()) {
      param += typeNameFor(arg);
      param += ", ";
    }
    return typeNameFor(fn_ty->getReturnType()) + "(" + param + ")";
  } else if (ty->isVoidTy()) {
    return "void";
  }
  assert(false);
}

InstrCount MemOpInstrumentation::instrumentStack(const StackArgList& stack) {
  using namespace transform;
  InstrCount counter{0};
  StackCounter::StackOpCounter allocCounts;
  Function* f{nullptr};
  for (const auto& [sdata, args] : stack) {
    auto* alloca = args.get_as<llvm::AllocaInst>(ArgMap::ID::pointer);
    auto& ctx    = alloca->getContext();

    // Insert the PointerInfo struct
    auto info_type = alloca->getModule()->getTypeByName("Typeart_PointerInfo");
    if (info_type == nullptr) {
      info_type = llvm::StructType::create({instr_helper->getTypeFor(IType::type_id),
                                            instr_helper->getTypeFor(IType::extent), llvm::Type::getInt64Ty(ctx)},
                                           "Typeart_PointerInfo");
    }

    // Create a struct data type for the stack allocation
    auto allocated_type = alloca->getAllocatedType();
    auto type           = (llvm::Type*)nullptr;
    auto name           = std::string{"Typeart_Wrapper_"};  // + typeNameFor(allocated_type);
    // type                = alloca->getModule()->getTypeByName(name);
    if (type == nullptr) {
      type = llvm::StructType::create({info_type, llvm::Type::getInt64Ty(ctx), alloca->getAllocatedType()}, name);
    }

    IRBuilder<> IRB(alloca->getNextNode());

    auto typeIdConst    = args.get_value(ArgMap::ID::type_id);
    auto numElementsVal = args.get_value(ArgMap::ID::element_count);

    auto parent_f          = sdata.alloca->getFunction();
    const auto callback_id = util::omp::isOmpContext(parent_f) ? IFunc::stack_omp : IFunc::stack;
    auto instr_alloca      = IRB.CreateAlloca(type);
    instr_alloca->setAlignment(llvm::MaybeAlign{alloca->getAlignment()});
    auto ptr_info = IRB.CreateStructGEP(instr_alloca, 0);
    auto type_id  = IRB.CreateStructGEP(ptr_info, 0);
    IRB.CreateStore(typeIdConst, type_id);
    auto count = IRB.CreateStructGEP(ptr_info, 1);
    IRB.CreateStore(numElementsVal, count);
    auto parity = IRB.CreateStructGEP(ptr_info, 2);
    IRB.CreateStore(llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx), parity_constant), parity);
    auto user_data = IRB.CreateStructGEP(instr_alloca, 2);
    alloca->replaceAllUsesWith(user_data);

    // TODO
    auto arrayPtr = IRB.CreateBitOrPointerCast(user_data, instr_helper->getTypeFor(IType::ptr));
    IRB.CreateCall(fquery->getFunctionFor(callback_id), ArrayRef<Value*>{arrayPtr, typeIdConst, numElementsVal});

    ++counter;

    auto bb = alloca->getParent();
    allocCounts[bb]++;
    if (!f) {
      f = bb->getParent();
    }

    // alloca->removeFromParent();
  }

  if (f) {
    StackCounter scount(f, instr_helper, fquery);
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
      auto& ctx           = global->getContext();
      auto typeIdConst    = args.get_value(ArgMap::ID::type_id);
      auto numElementsVal = args.get_value(ArgMap::ID::element_count);
      // auto globalPtr      = IRB.CreateBitOrPointerCast(global, instr_helper->getTypeFor(IType::ptr));
      // IRB.CreateCall(fquery->getFunctionFor(IFunc::global), ArrayRef<Value*>{globalPtr, typeIdConst,
      // numElementsVal});

      // auto ptr_info = IRB.CreateStructGEP(global, 0);
      // auto type_id  = IRB.CreateStructGEP(ptr_info, 0);
      // IRB.CreateStore(typeIdConst, type_id);
      // auto count = IRB.CreateStructGEP(ptr_info, 1);
      // IRB.CreateStore(numElementsVal, count);
      // auto addr    = IRB.CreateStructGEP(ptr_info, 2);
      // auto i8_null = llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(ctx)));
      // IRB.CreateStore(i8_null, addr);
      // auto user_data = IRB.CreateStructGEP(global, 1);
      // global->replaceAllUsesWith(user_data);
      // global->removeFromParent();

      ++counter;
    }
  };

  const auto makeCtorFuncBody = [&]() -> IRBuilder<> {
    auto m                = instr_helper->getModule();
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
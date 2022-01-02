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

#include "MemOpArgCollector.h"

#include "Instrumentation.h"
#include "InstrumentationHelper.h"
#include "support/Logger.h"
#include "support/TypeUtil.h"
#include "support/Util.h"
#include "typegen/TypeGenerator.h"
#include "typelib/TypeInterface.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstddef>

namespace llvm {
class DataLayout;
class Value;
}  // namespace llvm

namespace tu = typeart::util::type;
using namespace llvm;

namespace typeart {

MemOpArgCollector::MemOpArgCollector(TypeGenerator* tm, InstrumentationHelper& instr)
    : ArgumentCollector(), type_m(tm), instr_helper(&instr) {
}

HeapArgList MemOpArgCollector::collectHeap(const MallocDataList& mallocs) {
  HeapArgList list;
  list.reserve(mallocs.size());
  const llvm::DataLayout& dl = instr_helper->getModule()->getDataLayout();
  for (const MallocData& mdata : mallocs) {
    ArgMap arg_map;
    const auto malloc_call      = mdata.call;
    Value* pointer              = malloc_call;
    BitCastInst* primaryBitcast = mdata.primary;
    auto kind                   = mdata.kind;

    // Number of bytes allocated
    auto mallocArg = malloc_call->getOperand(0);
    int typeId     = type_m->getOrRegisterType(malloc_call->getType()->getPointerElementType(),
                                           dl);  // retrieveTypeID(tu::getVoidType(c));
    if (typeId == TYPEART_UNKNOWN_TYPE) {
      LOG_ERROR("Unknown heap type. Not instrumenting. " << util::dump(*malloc_call));
      // TODO notify caller that we skipped: via lambda callback function
      continue;
    }

    // Number of bytes per element, 1 for void*
    unsigned typeSize = tu::getTypeSizeInBytes(malloc_call->getType()->getPointerElementType(), dl);

    // Use the first cast as the determining type (if there is any)
    if (primaryBitcast != nullptr) {
      auto* dstPtrType = primaryBitcast->getDestTy()->getPointerElementType();

      typeSize = tu::getTypeSizeInBytes(dstPtrType, dl);

      // Resolve arrays
      // TODO: Write tests for this case
      if (dstPtrType->isArrayTy()) {
        dstPtrType = tu::getArrayElementType(dstPtrType);
      }

      typeId = type_m->getOrRegisterType(dstPtrType, dl);
      if (typeId == TYPEART_UNKNOWN_TYPE) {
        LOG_ERROR("Target type of casted allocation is unknown. Not instrumenting. " << util::dump(*malloc_call));
        LOG_ERROR("Cast: " << util::dump(*primaryBitcast));
        LOG_ERROR("Target type: " << util::dump(*dstPtrType));
        // TODO notify caller that we skipped: via lambda callback function
        continue;
      }
    } else {
      LOG_WARNING("Primary bitcast is null. malloc: " << util::dump(*malloc_call))
    }

    auto* typeIdConst    = instr_helper->getConstantFor(IType::type_id, typeId);
    Value* typeSizeConst = instr_helper->getConstantFor(IType::extent, typeSize);

    Value* elementCount{nullptr};
    Value* byte_count{nullptr};
    Value* realloc_ptr{nullptr};
    switch (kind) {
      case MemOpKind::NewLike:
        [[fallthrough]];
      case MemOpKind::MallocLike:
        if (mdata.array_cookie.hasValue()) {
          auto array_cookie_data = mdata.array_cookie.getValue();
          elementCount           = array_cookie_data.cookie_store->getValueOperand();
          pointer                = array_cookie_data.array_ptr_gep;
        }

        byte_count = mallocArg;
        break;
      case MemOpKind::CallocLike: {
        if (mdata.primary == nullptr) {
          // we need the second arg when the calloc type is identified as void* to calculate total bytes allocated
          typeSizeConst = malloc_call->getOperand(1);
        }
        elementCount = malloc_call->getOperand(0);
        break;
      }
      case MemOpKind::ReallocLike:
        realloc_ptr = malloc_call->getOperand(0);
        byte_count  = malloc_call->getOperand(1);
        break;
      case MemOpKind::AlignedAllocLike:
        byte_count = malloc_call->getArgOperand(1);
        break;
      default:
        LOG_ERROR("Unknown malloc kind. Not instrumenting. " << util::dump(*malloc_call));
        // TODO see above continues
        continue;
    }

    arg_map[ArgMap::ID::pointer]       = pointer;
    arg_map[ArgMap::ID::type_id]       = typeIdConst;
    arg_map[ArgMap::ID::type_size]     = typeSizeConst;
    arg_map[ArgMap::ID::byte_count]    = byte_count;
    arg_map[ArgMap::ID::element_count] = elementCount;
    arg_map[ArgMap::ID::realloc_ptr]   = realloc_ptr;
    list.emplace_back(HeapArgList::value_type{mdata, arg_map});
  }

  return list;
}

FreeArgList MemOpArgCollector::collectFree(const FreeDataList& frees) {
  FreeArgList list;
  list.reserve(frees.size());
  for (const FreeData& fdata : frees) {
    ArgMap arg_map;
    auto free_call = fdata.call;

    Value* freeArg{nullptr};
    switch (fdata.kind) {
      case MemOpKind::DeleteLike:
        [[fallthrough]];
      case MemOpKind::FreeLike:
        freeArg = fdata.array_cookie_gep.hasValue() ? fdata.array_cookie_gep.getValue()->getPointerOperand()
                                                    : free_call->getOperand(0);
        break;
      default:
        LOG_ERROR("Unknown free kind. Not instrumenting. " << util::dump(*free_call));
        continue;
    }

    arg_map[ArgMap::ID::pointer] = freeArg;
    list.emplace_back(FreeArgList::value_type{fdata, arg_map});
  }

  return list;
}

StackArgList MemOpArgCollector::collectStack(const AllocaDataList& allocs) {
  using namespace llvm;
  StackArgList list;
  list.reserve(allocs.size());
  const llvm::DataLayout& dl = instr_helper->getModule()->getDataLayout();

  for (const AllocaData& adata : allocs) {
    ArgMap arg_map;
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
      numElementsVal = instr_helper->getConstantFor(IType::extent, arraySize);
    }

    // unsigned typeSize = tu::getTypeSizeInBytes(elementType, dl);
    int typeId = type_m->getOrRegisterType(elementType, dl);

    if (typeId == TYPEART_UNKNOWN_TYPE) {
      LOG_ERROR("Unknown stack type. Not instrumenting. " << util::dump(*elementType));
      continue;
    }

    auto* typeIdConst = instr_helper->getConstantFor(IType::type_id, typeId);

    arg_map[ArgMap::ID::pointer]       = alloca;
    arg_map[ArgMap::ID::type_id]       = typeIdConst;
    arg_map[ArgMap::ID::element_count] = numElementsVal;

    list.emplace_back(StackArgList::value_type{adata, arg_map});
  }

  return list;
}

GlobalArgList MemOpArgCollector::collectGlobal(const GlobalDataList& globals) {
  GlobalArgList list;
  list.reserve(globals.size());
  const llvm::DataLayout& dl = instr_helper->getModule()->getDataLayout();

  for (const GlobalData& gdata : globals) {
    ArgMap arg_map;
    auto global = gdata.global;
    auto type   = global->getValueType();

    unsigned numElements = 1;
    if (type->isArrayTy()) {
      numElements = tu::getArrayLengthFlattened(type);
      type        = tu::getArrayElementType(type);
    }

    int typeId = type_m->getOrRegisterType(type, dl);

    if (typeId == TYPEART_UNKNOWN_TYPE) {
      LOG_ERROR("Unknown global type. Not instrumenting. " << util::dump(*type));
      continue;
    }

    auto* typeIdConst      = instr_helper->getConstantFor(IType::type_id, typeId);
    auto* numElementsConst = instr_helper->getConstantFor(IType::extent, numElements);
    // auto globalPtr         = IRB.CreateBitOrPointerCast(global, instr.getTypeFor(IType::ptr));

    arg_map[ArgMap::ID::pointer]       = global;
    arg_map[ArgMap::ID::type_id]       = typeIdConst;
    arg_map[ArgMap::ID::element_count] = numElementsConst;

    list.emplace_back(GlobalArgList::value_type{gdata, arg_map});
  }

  return list;
}

}  // namespace typeart
// TypeART library
//
// Copyright (c) 2017-2025 TypeART Authors
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

  for (const MallocData& mdata : mallocs) {
    ArgMap arg_map;
    const auto malloc_call = mdata.call;

    const auto [type_id, num_elements] = type_m->getOrRegisterType(mdata);

    if (type_id == TYPEART_UNKNOWN_TYPE) {
      LOG_DEBUG("Target type of casted allocation is unknown. Not instrumenting. " << util::dump(*malloc_call));
      continue;
    }

    auto type_size = type_m->getTypeDatabase().getTypeSize(type_id);
    if (type_id == TYPEART_VOID) {
      type_size = 1;
    }

    LOG_DEBUG("Type " << type_id << " with " << type_size << " and num elems " << num_elements)

    auto* type_id_const    = instr_helper->getConstantFor(IType::type_id, type_id);
    Value* type_size_const = instr_helper->getConstantFor(IType::extent, type_size);

    Value* element_count{nullptr};
    Value* byte_count{nullptr};
    Value* realloc_ptr{nullptr};
    Value* pointer = malloc_call;

    switch (mdata.kind) {
      case MemOpKind::NewLike:
        [[fallthrough]];
      case MemOpKind::MallocLike:
        if (mdata.array_cookie) {
          auto array_cookie_data = mdata.array_cookie.value();
          element_count          = array_cookie_data.cookie_store->getValueOperand();
          pointer                = array_cookie_data.array_ptr_gep;
        }

        byte_count = malloc_call->getOperand(0);

        break;
      case MemOpKind::CallocLike: {
        if (mdata.primary == nullptr) {
          // we need the second arg when the calloc type is identified as void* to calculate total bytes allocated
          type_size_const = malloc_call->getOperand(1);
        }
        element_count = malloc_call->getOperand(0);
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
    arg_map[ArgMap::ID::type_id]       = type_id_const;
    arg_map[ArgMap::ID::type_size]     = type_size_const;
    arg_map[ArgMap::ID::byte_count]    = byte_count;
    arg_map[ArgMap::ID::element_count] = element_count;
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

    Value* free_arg{nullptr};
    switch (fdata.kind) {
      case MemOpKind::DeleteLike:
        [[fallthrough]];
      case MemOpKind::FreeLike:
        free_arg =
            fdata.array_cookie_gep ? fdata.array_cookie_gep.value()->getPointerOperand() : free_call->getOperand(0);
        break;
      default:
        LOG_ERROR("Unknown free kind. Not instrumenting. " << util::dump(*free_call));
        continue;
    }

    arg_map[ArgMap::ID::pointer] = free_arg;
    list.emplace_back(FreeArgList::value_type{fdata, arg_map});
  }

  return list;
}

StackArgList MemOpArgCollector::collectStack(const AllocaDataList& allocs) {
  using namespace llvm;
  StackArgList list;
  list.reserve(allocs.size());

  for (const AllocaData& adata : allocs) {
    ArgMap arg_map;
    auto alloca = adata.alloca;

    const auto [type_id, num_elements] = type_m->getOrRegisterType(adata);

    if (type_id == TYPEART_UNKNOWN_TYPE) {
      LOG_DEBUG("Unknown stack type. Not instrumenting. " << util::dump(*alloca));
      continue;
    }

    auto type_size = type_m->getTypeDatabase().getTypeSize(type_id);
    if (type_id == TYPEART_VOID) {
      type_size = 1;
    }

    LOG_DEBUG("Alloca Type " << type_id << " with " << type_size << " and num elems " << num_elements)

    Value* num_elements_val{nullptr};
    // The length can be specified statically through the array type or as a separate argument.
    // Both cases are handled here.
    if (adata.is_vla) {
      LOG_DEBUG("Found VLA array allocation")
      // This should not happen in generated IR code
      assert(!alloca->getAllocatedType()->isArrayTy() && "VLAs of array types are currently not supported.");
      num_elements_val = alloca->getArraySize();
    } else {
      num_elements_val = instr_helper->getConstantFor(IType::extent, num_elements);
    }

    auto* type_id_constant = instr_helper->getConstantFor(IType::type_id, type_id);

    arg_map[ArgMap::ID::pointer]       = alloca;
    arg_map[ArgMap::ID::type_id]       = type_id_constant;
    arg_map[ArgMap::ID::element_count] = num_elements_val;

    list.emplace_back(StackArgList::value_type{adata, arg_map});
  }

  return list;
}

GlobalArgList MemOpArgCollector::collectGlobal(const GlobalDataList& globals) {
  GlobalArgList list;
  list.reserve(globals.size());

  for (const GlobalData& gdata : globals) {
    ArgMap arg_map;
    auto global = gdata.global;

    const auto [type_id, num_elements] = type_m->getOrRegisterType(gdata);

    if (type_id == TYPEART_UNKNOWN_TYPE) {
      LOG_DEBUG("Unknown global type. Not instrumenting. " << util::dump(*global));
      continue;
    }

    auto type_size = type_m->getTypeDatabase().getTypeSize(type_id);
    if (type_id == TYPEART_VOID) {
      type_size = 1;
    }

    LOG_DEBUG("Global Type " << type_id << " with " << type_size << " and num elems " << num_elements)

    auto* type_id_const      = instr_helper->getConstantFor(IType::type_id, type_id);
    auto* num_elements_const = instr_helper->getConstantFor(IType::extent, num_elements);
    // auto globalPtr         = IRB.CreateBitOrPointerCast(global, instr.getTypeFor(IType::ptr));

    arg_map[ArgMap::ID::pointer]       = global;
    arg_map[ArgMap::ID::type_id]       = type_id_const;
    arg_map[ArgMap::ID::element_count] = num_elements_const;

    list.emplace_back(GlobalArgList::value_type{gdata, arg_map});
  }

  return list;
}

}  // namespace typeart

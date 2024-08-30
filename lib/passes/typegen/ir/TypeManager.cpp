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

#include "TypeManager.h"

#include "IRTypeGen.h"
#include "StructTypeHandler.h"
#include "VectorTypeHandler.h"
#include "support/Logger.h"
#include "support/TypeUtil.h"
#include "support/Util.h"
#include "typegen/TypeIDGenerator.h"
#include "typelib/TypeInterface.h"

#include "llvm/ADT/None.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace typeart {

namespace tu = typeart::util::type;

std::unique_ptr<TypeGenerator> make_ir_typeidgen(std::string_view file) {
  return std::make_unique<TypeManager>(std::string{file});
}

using namespace llvm;

llvm::Optional<typeart_builtin_type> get_builtin_typeid(llvm::Type* type) {
  auto& c = type->getContext();

  switch (type->getTypeID()) {
    case llvm::Type::IntegerTyID: {
      if (type == Type::getInt8Ty(c)) {
        return TYPEART_INT8;
      }
      if (type == Type::getInt16Ty(c)) {
        return TYPEART_INT16;
      }
      if (type == Type::getInt32Ty(c)) {
        return TYPEART_INT32;
      }
      if (type == Type::getInt64Ty(c)) {
        return TYPEART_INT64;
      }
      return TYPEART_UNKNOWN_TYPE;
    }
    case llvm::Type::HalfTyID:
      return TYPEART_HALF;
    case llvm::Type::FloatTyID:
      return TYPEART_FLOAT;
    case llvm::Type::DoubleTyID:
      return TYPEART_DOUBLE;
    case llvm::Type::FP128TyID:
      return TYPEART_FP128;
    case llvm::Type::X86_FP80TyID:
      return TYPEART_X86_FP80;
    case llvm::Type::PPC_FP128TyID:
      return TYPEART_PPC_FP128;
    case llvm::Type::PointerTyID:
      return TYPEART_POINTER;
    default:
      return None;
  }
}

int TypeManager::getOrRegisterVector(llvm::VectorType* type, const llvm::DataLayout& dl) {
  namespace tu = typeart::util;

  VectorTypeHandler handler{&structMap, &typeDB, type, dl, *this};
  const auto type_id = handler.getID();
  if (type_id) {
    return type_id.getValue();
  }

  // Type is not registered - reserve new ID and create struct info object:

  auto element_data = handler.getElementData();
  if (!element_data) {
    return TYPEART_UNKNOWN_TYPE;
  }

  auto vector_data = handler.getVectorData();
  if (!vector_data) {
    return TYPEART_UNKNOWN_TYPE;
  }

  const int id = reserveNextTypeId();

  const auto [element_id, element_type, element_name] = element_data.getValue();
  const auto [vector_name, vector_bytes, vector_size] = vector_data.getValue();

  std::vector<int> memberTypeIDs;
  std::vector<size_t> arraySizes;
  std::vector<size_t> offsets;

  size_t elementSize = tu::type::getTypeSizeInBytes(element_type, dl);

  for (int i = 0; i < vector_size; ++i) {
    memberTypeIDs.push_back(element_id);
    arraySizes.push_back(1);
    offsets.push_back(i * elementSize);
  }

  // size_t usableBytes = vector_size * elementSize;

  // Add padding bytes explicitly
  // if (vector_bytes > usableBytes) {
  //   size_t padding = vector_bytes - usableBytes;
  //   memberTypeIDs.push_back(TYPEART_INT8);
  //   arraySizes.push_back(padding);
  //   offsets.push_back(usableBytes);
  // }

  StructTypeInfo vecTypeInfo{id,      vector_name,   vector_bytes, memberTypeIDs.size(),
                             offsets, memberTypeIDs, arraySizes,   StructTypeFlag::LLVM_VECTOR};
  typeDB.registerStruct(vecTypeInfo);
  structMap.insert({vector_name, id});
  return id;
}

TypeManager::TypeManager(std::string file) : types::TypeIDGenerator(std::move(file)) {
}

int TypeManager::getTypeID(llvm::Type* type, const DataLayout& dl) const {
  auto builtin_id = get_builtin_typeid(type);
  if (builtin_id) {
    return builtin_id.getValue();
  }

  switch (type->getTypeID()) {
#if LLVM_VERSION_MAJOR < 11
    case llvm::Type::VectorTyID:
#else
    case llvm::Type::FixedVectorTyID:
#endif
    {
      VectorTypeHandler handle{&structMap, &typeDB, dyn_cast<VectorType>(type), dl, *this};
      const auto type_id = handle.getID();
      if (type_id) {
        return type_id.getValue();
      }
      break;
    }
    case llvm::Type::StructTyID: {
      StructTypeHandler handle{&structMap, &typeDB, dyn_cast<StructType>(type)};
      const auto type_id = handle.getID();
      if (type_id) {
        return type_id.getValue();
      }
      break;
    }
#if LLVM_VERSION_MAJOR > 10
    case llvm::Type::ScalableVectorTyID: {
      LOG_ERROR("Scalable SIMD vectors are unsupported.")
      break;
    }
#endif
    default:
      break;
  }

  return TYPEART_UNKNOWN_TYPE;
}

int TypeManager::getOrRegisterType(llvm::Type* type, const llvm::DataLayout& dl) {
  auto builtin_id = get_builtin_typeid(type);
  if (builtin_id) {
    return builtin_id.getValue();
  }

  switch (type->getTypeID()) {
#if LLVM_VERSION_MAJOR < 11
    case llvm::Type::VectorTyID:
#else
    case llvm::Type::FixedVectorTyID:
#endif
    {
      return getOrRegisterVector(dyn_cast<VectorType>(type), dl);
    }
    case llvm::Type::StructTyID:
      return getOrRegisterStruct(dyn_cast<StructType>(type), dl);
    default:
      break;
  }
  return TYPEART_UNKNOWN_TYPE;
}

int TypeManager::getOrRegisterStruct(llvm::StructType* type, const llvm::DataLayout& dl) {
  namespace tu = typeart::util;

  StructTypeHandler handle{&structMap, &typeDB, type};
  const auto type_id = handle.getID();
  if (type_id) {
    return type_id.getValue();
  }

  const auto name = handle.getName();

  // Get next ID and register struct:
  const int id = reserveNextTypeId();

  size_t n = type->getStructNumElements();

  std::vector<size_t> offsets;
  std::vector<size_t> arraySizes;
  std::vector<int> memberTypeIDs;

  const StructLayout* layout = dl.getStructLayout(type);

  for (unsigned i = 0; i < n; ++i) {
    llvm::Type* memberType = type->getStructElementType(i);
    int memberID           = TYPEART_UNKNOWN_TYPE;
    size_t arraySize       = 1;

    if (memberType->isArrayTy()) {
      // Note that clang does not allow VLAs inside of structs (GCC does)
      arraySize  = tu::type::getArrayLengthFlattened(memberType);
      memberType = tu::type::getArrayElementType(memberType);
    }

    if (memberType->isStructTy()) {
      if (StructTypeHandler::getName(llvm::dyn_cast<StructType>(memberType)) == name) {
        memberID = id;
      } else {
        memberID = getOrRegisterType(memberType, dl);
      }
    } else if (memberType->isSingleValueType() || memberType->isPointerTy()) {
      memberID = getOrRegisterType(memberType, dl);
    } else {
      // clang-format off
      LOG_ERROR("In struct: " << tu::dump(*type)
                  << ": Encountered unhandled type: " << tu::dump(*memberType)
                  << " with type id: " << memberType->getTypeID());
      // clang-format on
      assert(false && "Encountered unhandled type");
    }

    size_t offset = layout->getElementOffset(i);
    offsets.push_back(offset);
    arraySizes.push_back(arraySize);
    memberTypeIDs.push_back(memberID);
  }

  size_t numBytes = layout->getSizeInBytes();

  StructTypeInfo structInfo{id, name, numBytes, n, offsets, memberTypeIDs, arraySizes, StructTypeFlag::USER_DEFINED};
  typeDB.registerStruct(structInfo);

  structMap.insert({name, id});
  return id;
}

TypeIdentifier TypeManager::getOrRegisterType(const MallocData& mdata) {
  auto malloc_call            = mdata.call;
  const llvm::DataLayout& dl  = malloc_call->getModule()->getDataLayout();
  Value* pointer              = malloc_call;
  BitCastInst* primaryBitcast = mdata.primary;
  auto kind                   = mdata.kind;

  auto mallocArg = malloc_call->getOperand(0);
  int typeId     = getOrRegisterType(malloc_call->getType()->getPointerElementType(),
                                     dl);  // retrieveTypeID(tu::getVoidType(c));
  if (typeId == TYPEART_UNKNOWN_TYPE) {
    LOG_ERROR("Unknown heap type. Not instrumenting. " << util::dump(*malloc_call));
    // TODO notify caller that we skipped: via lambda callback function
    return {};
  };

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

    typeId = getOrRegisterType(dstPtrType, dl);
    if (typeId == TYPEART_UNKNOWN_TYPE) {
      LOG_ERROR("Target type of casted allocation is unknown. Not instrumenting. " << util::dump(*malloc_call));
      LOG_ERROR("Cast: " << util::dump(*primaryBitcast));
      LOG_ERROR("Target type: " << util::dump(*dstPtrType));
    }
  } else {
    LOG_WARNING("Primary bitcast is null. malloc: " << util::dump(*malloc_call))
  }
  return {typeId, 0};
}

TypeIdentifier TypeManager::getOrRegisterType(const AllocaData& adata) {
  auto alloca                = adata.alloca;
  const llvm::DataLayout& dl = alloca->getModule()->getDataLayout();
  Type* elementType          = alloca->getAllocatedType();

  auto arraySize = adata.array_size == 0 ? 1 : adata.array_size;
  if (elementType->isArrayTy()) {
    arraySize   = arraySize * tu::getArrayLengthFlattened(elementType);
    elementType = tu::getArrayElementType(elementType);
  }

  int typeId = getOrRegisterType(elementType, dl);

  return {typeId, arraySize};
}

TypeIdentifier TypeManager::getOrRegisterType(const GlobalData& gdata) {
  auto global                = gdata.global;
  const llvm::DataLayout& dl = global->getParent()->getDataLayout();
  auto type                  = global->getValueType();

  size_t num_elements{1};
  if (type->isArrayTy()) {
    num_elements = tu::getArrayLengthFlattened(type);
    type         = tu::getArrayElementType(type);
  }

  int typeId = getOrRegisterType(type, dl);
  return {typeId, num_elements};
}

}  // namespace typeart

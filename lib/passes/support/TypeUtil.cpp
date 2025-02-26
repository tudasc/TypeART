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

#include "TypeUtil.h"

#include "support/Logger.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace typeart::util::type {

#if LLVM_VERSION_MAJOR < 15
bool isi64Ptr(llvm::Type* type) {
  return type->isPointerTy() && type->getPointerElementType()->isIntegerTy(64);
}

bool isVoidPtr(llvm::Type* type) {
  return type->isPointerTy() && type->getPointerElementType()->isIntegerTy(8);
}
#endif

/**
 * Code was imported from jplehr/llvm-memprofiler project
 */
unsigned getTypeSizeInBytes(llvm::Type* t, const llvm::DataLayout& dl) {
  unsigned bytes = getScalarSizeInBytes(t);

  if (t->isArrayTy()) {
    bytes = getArraySizeInBytes(t, dl);
  } else if (t->isStructTy()) {
    bytes = getStructSizeInBytes(t, dl);
  } else if (t->isPointerTy()) {
    bytes = getPointerSizeInBytes(t, dl);
  } else if (t->isVectorTy()) {
    bytes = getVectorSizeInBytes(t, dl);
  }

  return bytes;
}

unsigned getScalarSizeInBytes(llvm::Type* t) {
  return t->getScalarSizeInBits() / 8;
}

unsigned getArraySizeInBytes(llvm::Type* arrT, const llvm::DataLayout& dl) {
  auto st = dyn_cast<ArrayType>(arrT);
  return getTypeSizeInBytes(getArrayElementType(st), dl) * getArrayLengthFlattened(st);
}

unsigned getVectorSizeInBytes(llvm::Type* vectorT, const llvm::DataLayout& dl) {
  // TODO: Most of these utility functions can be eliminated with the use of getTypeAllocSize() and getTypeStoreSize()
  //  auto vt = dyn_cast<VectorType>(vectorT);
  return dl.getTypeAllocSize(vectorT);
  // return getTypeSizeInBytes(vt->getVectorElementType(), dl) * vt->getVectorNumElements();
}

/**
 * \brief Resolves the element type of the given array recursively. Works for multidimensional arrays.
 */
llvm::Type* getArrayElementType(llvm::Type* arrT) {
  while (arrT->isArrayTy()) {
    arrT = arrT->getArrayElementType();
  }
  return arrT;
}

/**
 * \brief Determines the length of the flattened array.
 *  TODO: Handle VLAs
 */
unsigned getArrayLengthFlattened(llvm::Type* arrT) {
  unsigned len = 1;
  while (arrT->isArrayTy()) {
    len *= arrT->getArrayNumElements();
    arrT = arrT->getArrayElementType();
  }
  return len;
}

unsigned getStructSizeInBytes(llvm::Type* structT, const llvm::DataLayout& dl) {
  auto st                    = dyn_cast<llvm::StructType>(structT);
  const StructLayout* layout = dl.getStructLayout(st);
  return layout->getSizeInBytes();
}

unsigned getPointerSizeInBytes(llvm::Type* /*ptrT*/, const llvm::DataLayout& dl) {
  return dl.getPointerSizeInBits() / 8;
}

}  // namespace typeart::util::type

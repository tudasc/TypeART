#include "TypeUtil.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"

#include "Logger.h"
#include <iostream>

using namespace llvm;

namespace util {
namespace type {

Type* getVoidType(LLVMContext& c) {
  return Type::getVoidTy(c);
}

Type* getVoidPtrType(LLVMContext& c) {
  // NOTE: LLVM does not support an explicit void pointer
  return Type::getInt8PtrTy(c);
}

Type* getInt32Type(LLVMContext& c) {
  return Type::getInt32Ty(c);
}

Type* getInt64Type(LLVMContext& c) {
  return Type::getInt64Ty(c);
}

bool isVoidPtr(llvm::Type* type) {
  return type->isPointerTy() && type->getPointerElementType()->isIntegerTy(8);
}

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
  }

  return bytes;
}

unsigned getScalarSizeInBytes(llvm::Type* t) {
  return t->getScalarSizeInBits() / 8;
}

unsigned getArraySizeInBytes(llvm::Type* arrT, const llvm::DataLayout& dl) {
  auto st = dyn_cast<ArrayType>(arrT);
  Type* underlyingType = st->getElementType();
  unsigned bytes = getScalarSizeInBytes(underlyingType);
  bytes *= st->getNumElements();
  std::cout << "Determined number of bytes to allocate: " << bytes << std::endl;

  return bytes;
}

unsigned getStructSizeInBytes(llvm::Type* structT, const llvm::DataLayout& dl) {
  auto st = dyn_cast<llvm::StructType>(structT);
  const StructLayout* layout = dl.getStructLayout(st);
  return layout->getSizeInBytes();
}

unsigned getPointerSizeInBytes(llvm::Type* ptrT, const llvm::DataLayout& dl) {
  return dl.getPointerSizeInBits() / 8;
}

unsigned getTypeSizeForArrayAlloc(llvm::AllocaInst* ai, const llvm::DataLayout& dl) {
  auto type = ai->getAllocatedType();
  unsigned bytes;
  if (type->isArrayTy()) {
    bytes = getTypeSizeInBytes(type->getArrayElementType(), dl);
  } else {
    bytes = getTypeSizeInBytes(type, dl);
  }
  if (ai->isArrayAllocation()) {
    if (auto ci = dyn_cast<ConstantInt>(ai->getArraySize())) {
      bytes *= ci->getLimitedValue();
    } else {
      // If this can not be determined statically, we have to compute it at
      // runtime. We insert additional instructions to calculate the
      // numBytes of that array on the fly. (VLAs produce this kind of
      // behavior)
      // ATTENTION: We can have multiple such arrays in a single BB. We need
      // to have a small vector to store whether we already generated
      // instructions, to possibly refer to the results for further
      // calculations.
      std::cout << "We hit not yet determinable array size expression\n";
    }
  }
  return bytes;
}

}  // namespace type
}  // namespace util

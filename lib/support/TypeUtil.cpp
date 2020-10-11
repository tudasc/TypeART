#include "TypeUtil.h"

#include "Logger.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"

using namespace llvm;

namespace typeart {
namespace util {
namespace type {

bool isi64Ptr(llvm::Type* type) {
  return type->isPointerTy() && type->getPointerElementType()->isIntegerTy(64);
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

unsigned getPointerSizeInBytes(llvm::Type* ptrT, const llvm::DataLayout& dl) {
  return dl.getPointerSizeInBits() / 8;
}

unsigned getTypeSizeForArrayAlloc(llvm::AllocaInst* ai, const llvm::DataLayout& dl) {
  // TODO: Not used at the moment. Integrate VLA check into replacement methods (getArrayLengthFlattened).

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
      LOG_WARNING("We hit not yet determinable array size expression: " << *ai);
    }
  }
  return bytes;
}

}  // namespace type
}  // namespace util
}  // namespace typeart

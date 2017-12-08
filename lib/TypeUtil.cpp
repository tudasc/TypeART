#include "TypeUtil.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Instructions.h"


using namespace llvm;

namespace util {
	namespace type {

int getTypeSizeInBytes(llvm::Type *t, const llvm::DataLayout &dl){}

int getScalarSizeInBytes(llvm::Type *t){}

int getArraySizeInBytes(llvm::Type *arrT, const llvm::DataLayout &dl){}

int getStructSizeInBytes(llvm::Type *structT, const llvm::DataLayout &dl){}

int getPointerSizeInBytes(llvm::Type *ptrT, const llvm::DataLayout &dl){}

int getTypeSizeForArrayAlloc(llvm::AllocaInst *ai, const llvm::DataLayout &dl){}

	} // type
} // util 

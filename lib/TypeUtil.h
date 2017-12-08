#ifndef LIB_UTIL_TYPE_H
#define LIB_UTIL_TYPE_H

namespace llvm {
	class DataLayout;
	class Type;
	class AllocaInst;
}

namespace util {
	namespace type {

int getTypeSizeInBytes(llvm::Type *t, const llvm::DataLayout &dl);

int getScalarSizeInBytes(llvm::Type *t);

int getArraySizeInBytes(llvm::Type *arrT, const llvm::DataLayout &dl);

int getStructSizeInBytes(llvm::Type *structT, const llvm::DataLayout &dl);

int getPointerSizeInBytes(llvm::Type *ptrT, const llvm::DataLayout &dl);

int getTypeSizeForArrayAlloc(llvm::AllocaInst *ai, const llvm::DataLayout &dl);

	} // type
} // util

#endif

#ifndef LIB_UTIL_TYPE_H
#define LIB_UTIL_TYPE_H

namespace llvm {
class DataLayout;
class Type;
class AllocaInst;
class LLVMContext;
}  // namespace llvm

namespace util {
namespace type {

llvm::Type* getVoidType(llvm::LLVMContext& c);
llvm::Type* getVoidPtrType(llvm::LLVMContext& c);
llvm::Type* getInt32Type(llvm::LLVMContext& c);
llvm::Type* getInt64Type(llvm::LLVMContext& c);

bool isVoidPtr(llvm::Type* type);

unsigned getTypeSizeInBytes(llvm::Type* t, const llvm::DataLayout& dl);

unsigned getScalarSizeInBytes(llvm::Type* t);

unsigned getArraySizeInBytes(llvm::Type* arrT, const llvm::DataLayout& dl);

unsigned getStructSizeInBytes(llvm::Type* structT, const llvm::DataLayout& dl);

unsigned getPointerSizeInBytes(llvm::Type* ptrT, const llvm::DataLayout& dl);

unsigned getTypeSizeForArrayAlloc(llvm::AllocaInst* ai, const llvm::DataLayout& dl);

}  // namespace type
}  // namespace util

#endif

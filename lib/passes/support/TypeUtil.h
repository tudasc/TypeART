// TypeART library
//
// Copyright (c) 2017-2026 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#ifndef TYPEART_TYPE_UTIL_H
#define TYPEART_TYPE_UTIL_H

namespace llvm {
class DataLayout;
class Type;
class AllocaInst;
class LLVMContext;
}  // namespace llvm

#include <optional>

namespace typeart::util::type {

#if LLVM_VERSION_MAJOR < 15
bool isi64Ptr(llvm::Type* type);

bool isVoidPtr(llvm::Type* type);
#endif

unsigned getTypeSizeInBytes(llvm::Type* t, const llvm::DataLayout& dl);

unsigned getScalarSizeInBytes(llvm::Type* t);

unsigned getArraySizeInBytes(llvm::Type* arrT, const llvm::DataLayout& dl);

unsigned getVectorSizeInBytes(llvm::Type* vectorT, const llvm::DataLayout& dl);

llvm::Type* getArrayElementType(llvm::Type* arrT);

unsigned getArrayLengthFlattened(llvm::Type* arrT);

unsigned getStructSizeInBytes(llvm::Type* structT, const llvm::DataLayout& dl);

unsigned getPointerSizeInBytes(llvm::Type* ptrT, const llvm::DataLayout& dl);

std::optional<llvm::Type*> getPointerElementType(llvm::Type* ptr_type);

}  // namespace typeart::util::type

#endif  // TYPEART_TYPE_UTIL_H

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

#ifndef LLVM_MUST_SUPPORT_TYPEMANAGER_H
#define LLVM_MUST_SUPPORT_TYPEMANAGER_H

#include "typegen/TypeIDGenerator.h"

#include <memory>
#include <string_view>

namespace llvm {
class DataLayout;
class StructType;
class Type;
class VectorType;
}  // namespace llvm

namespace typeart {

class TypeManager final : public types::TypeIDGenerator {
 public:
  using types::TypeIDGenerator::TypeIDGenerator;
  [[nodiscard]] int getOrRegisterType(llvm::Type* type, const llvm::DataLayout& dl);
  [[nodiscard]] int getTypeID(llvm::Type* type, const llvm::DataLayout& dl) const;
  [[nodiscard]] int getOrRegisterType(llvm::Value* type);
  [[nodiscard]] TypeIdentifier getOrRegisterType(const MallocData&) override;
  [[nodiscard]] TypeIdentifier getOrRegisterType(const AllocaData&) override;
  [[nodiscard]] TypeIdentifier getOrRegisterType(const GlobalData&) override;

 protected:
  [[nodiscard]] int getOrRegisterStruct(llvm::StructType* type, const llvm::DataLayout& dl);
  [[nodiscard]] int getOrRegisterVector(llvm::VectorType* type, const llvm::DataLayout& dl);
};

}  // namespace typeart

#endif  // LLVM_MUST_SUPPORT_TYPEMANAGER_H

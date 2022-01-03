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

#ifndef LLVM_MUST_SUPPORT_TYPEMANAGER_H
#define LLVM_MUST_SUPPORT_TYPEMANAGER_H

#include "TypeGenerator.h"
#include "typelib/TypeDB.h"

#include "llvm/ADT/StringMap.h"

#include <cstddef>
#include <string>

namespace llvm {
class DataLayout;
class StructType;
class Type;
class VectorType;
}  // namespace llvm

namespace typeart {

class TypeManager final : public TypeGenerator {
  std::string file;
  TypeDB typeDB;
  llvm::StringMap<int> structMap;
  size_t structCount;

 public:
  explicit TypeManager(std::string file);
  [[nodiscard]] std::pair<bool, std::error_code> load() override;
  [[nodiscard]] std::pair<bool, std::error_code> store() const override;
  [[nodiscard]] int getOrRegisterType(llvm::Type* type, const llvm::DataLayout& dl) override;
  [[nodiscard]] int getTypeID(llvm::Type* type, const llvm::DataLayout& dl) const override;
  [[nodiscard]] const TypeDatabase& getTypeDatabase() const override;

 private:
  [[nodiscard]] int getOrRegisterStruct(llvm::StructType* type, const llvm::DataLayout& dl);
  [[nodiscard]] int getOrRegisterVector(llvm::VectorType* type, const llvm::DataLayout& dl);
  [[nodiscard]] int reserveNextId();
};

}  // namespace typeart

#endif  // LLVM_MUST_SUPPORT_TYPEMANAGER_H

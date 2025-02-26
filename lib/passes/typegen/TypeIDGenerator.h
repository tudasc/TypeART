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

#ifndef TYPEART_TYPEIDGENERATOR_H
#define TYPEART_TYPEIDGENERATOR_H

#include "TypeGenerator.h"
// #include "typelib/TypeDB.h"
#include "typelib/TypeDatabase.h"

#include "llvm/ADT/StringMap.h"

#include <memory>
#include <string>

namespace typeart::types {

class TypeIDGenerator : public TypeGenerator {
 protected:
  std::string file;
  std::unique_ptr<TypeDatabase> typeDB;
  llvm::StringMap<int> structMap;
  size_t structCount{0};

 public:
  explicit TypeIDGenerator(std::string file_, std::unique_ptr<TypeDatabase> database_of_types);

  virtual void registerModule(const ModuleData&) override;

  [[nodiscard]] virtual const TypeDatabase& getTypeDatabase() const override;

  [[nodiscard]] virtual std::pair<bool, std::error_code> load() override;
  [[nodiscard]] virtual std::pair<bool, std::error_code> store() const override;

  virtual ~TypeIDGenerator() = default;

 protected:
  [[nodiscard]] virtual int reserveNextTypeId();
};

}  // namespace typeart::types

#endif
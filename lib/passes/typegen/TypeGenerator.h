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

#ifndef TYPEART_TYPEGENERATOR_H
#define TYPEART_TYPEGENERATOR_H

#include "TypeDatabase.h"
#include "analysis/MemOpData.h"

#include <llvm/IR/Value.h>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace llvm {
class Type;
class DataLayout;
}  // namespace llvm

namespace typeart {

enum class TypegenImplementation { IR, DIMETA };

class TypeGenerator {
 public:
  [[nodiscard]] virtual int getOrRegisterType(const MallocData&) = 0;
  [[nodiscard]] virtual int getOrRegisterType(const AllocaData&) = 0;
  [[nodiscard]] virtual int getOrRegisterType(const GlobalData&) = 0;

  [[nodiscard]] virtual const TypeDatabase& getTypeDatabase() const = 0;

  [[nodiscard]] virtual std::pair<bool, std::error_code> load()        = 0;
  [[nodiscard]] virtual std::pair<bool, std::error_code> store() const = 0;

  virtual ~TypeGenerator() = default;
};

// This doesn't immediately load the file, call load/store after
std::unique_ptr<TypeGenerator> make_typegen(std::string_view file, TypegenImplementation impl);

}  // namespace typeart

#endif  // TYPEART_TYPEGENERATOR_H

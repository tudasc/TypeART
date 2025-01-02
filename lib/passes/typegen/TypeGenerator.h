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

#ifndef TYPEART_TYPEGENERATOR_H
#define TYPEART_TYPEGENERATOR_H

#include "TypeDatabase.h"
#include "analysis/MemOpData.h"
#include "typelib/TypeInterface.h"

#include <cstddef>
#include <cstdint>
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

struct TypeIdentifier final {
  int type_id{TYPEART_UNKNOWN_TYPE};
  std::uint64_t num_elements{1};  // > 1 for array-like type allocation
};

struct ModuleData {
  llvm::Module* module;
};

class TypeGenerator {
 public:
  virtual void registerModule(const ModuleData&)                            = 0;
  [[nodiscard]] virtual TypeIdentifier getOrRegisterType(const MallocData&) = 0;
  [[nodiscard]] virtual TypeIdentifier getOrRegisterType(const AllocaData&) = 0;
  [[nodiscard]] virtual TypeIdentifier getOrRegisterType(const GlobalData&) = 0;

  [[nodiscard]] virtual const TypeDatabase& getTypeDatabase() const = 0;

  [[nodiscard]] virtual std::pair<bool, std::error_code> load()        = 0;
  [[nodiscard]] virtual std::pair<bool, std::error_code> store() const = 0;

  virtual ~TypeGenerator() = default;
};

// This doesn't immediately load the file, call load/store after
std::unique_ptr<TypeGenerator> make_typegen(std::string_view file, TypegenImplementation impl);

}  // namespace typeart

#endif  // TYPEART_TYPEGENERATOR_H

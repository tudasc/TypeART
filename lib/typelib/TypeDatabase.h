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

#ifndef TYPEART_TYPEDATABASE_H
#define TYPEART_TYPEDATABASE_H

#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace typeart {

enum class StructTypeFlag : int { USER_DEFINED = 1, LLVM_VECTOR = 2 };

struct StructTypeInfo {
  int type_id;
  std::string name;
  size_t extent;
  size_t num_members;
  std::vector<size_t> offsets;
  std::vector<int> member_types;
  std::vector<size_t> array_sizes;
  StructTypeFlag flag;
};

class TypeDatabase {
 public:
  virtual void registerStruct(const StructTypeInfo& struct_info) = 0;

  [[nodiscard]] virtual bool isUnknown(int type_id) const = 0;

  [[nodiscard]] virtual bool isValid(int type_id) const = 0;

  [[nodiscard]] virtual bool isReservedType(int type_id) const = 0;

  [[nodiscard]] virtual bool isBuiltinType(int type_id) const = 0;

  [[nodiscard]] virtual bool isStructType(int type_id) const = 0;

  [[nodiscard]] virtual bool isUserDefinedType(int type_id) const = 0;

  [[nodiscard]] virtual bool isVectorType(int type_id) const = 0;

  [[nodiscard]] virtual const std::string& getTypeName(int type_id) const = 0;

  [[nodiscard]] virtual const StructTypeInfo* getStructInfo(int type_id) const = 0;

  [[nodiscard]] virtual size_t getTypeSize(int type_id) const = 0;

  [[nodiscard]] virtual const std::vector<StructTypeInfo>& getStructList() const = 0;

  virtual ~TypeDatabase() = default;
};

std::pair<std::unique_ptr<TypeDatabase>, std::error_code> make_database(const std::string& file);

}  // namespace typeart

#endif  // TYPEART_TYPEDATABASE_H

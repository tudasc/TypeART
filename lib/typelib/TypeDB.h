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

#ifndef LLVM_MUST_SUPPORT_TYPECONFIG_H
#define LLVM_MUST_SUPPORT_TYPECONFIG_H

#include "TypeDatabase.h"
#include "TypeInterface.h"

#include <array>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace typeart {

namespace builtins {

struct BuiltInQuery {
 private:
  const std::array<const std::string, TYPEART_NUM_VALID_IDS> names;
  const std::array<size_t, TYPEART_NUM_VALID_IDS> sizes;

  template <typename U>
  static constexpr auto type_info(const U& type_data, int type_id) -> decltype(type_data[type_id]) {
    if (type_id < 0 || type_id >= TYPEART_NUM_VALID_IDS) {
      return type_data[TYPEART_UNKNOWN_TYPE];
    }
    return type_data[type_id];
  }

 public:
  BuiltInQuery();

  [[nodiscard]] constexpr size_t get_size(int type_id) const {
    return type_info(sizes, type_id);
  }

  [[nodiscard]] const std::string& get_name(int type_id) const {
    return type_info(names, type_id);
  }

  static constexpr bool is_builtin_type(int type_id) {
    return type_id > TYPEART_UNKNOWN_TYPE && type_id < TYPEART_NUM_VALID_IDS;
  }

  static constexpr bool is_reserved_type(int type_id) {
    return type_id < TYPEART_NUM_RESERVED_IDS;
  }

  static constexpr bool is_userdef_type(int type_id) {
    return type_id >= TYPEART_NUM_RESERVED_IDS;
  }

  static constexpr bool is_unknown_type(int type_id) {
    return type_id == TYPEART_UNKNOWN_TYPE;
  }
};

}  // namespace builtins

class TypeDB final : public TypeDatabase {
 public:
  void clear() override;

  void registerStruct(const StructTypeInfo& struct_type, bool overwrite = false) override;

  bool isUnknown(int type_id) const override;

  bool isValid(int type_id) const override;

  bool isReservedType(int type_id) const override;

  bool isBuiltinType(int type_id) const override;

  bool isStructType(int type_id) const override;

  bool isUserDefinedType(int type_id) const override;

  bool isVectorType(int type_id) const override;

  const std::string& getTypeName(int type_id) const override;

  const StructTypeInfo* getStructInfo(int type_id) const override;
  [[nodiscard]] StructTypeInfo* getStructInfo(int type_id) override;

  size_t getTypeSize(int type_id) const override;

  const std::vector<StructTypeInfo>& getStructList() const override;

 private:
  builtins::BuiltInQuery builtins;
  std::vector<StructTypeInfo> struct_info_vec;
  std::unordered_map<int, int> typeid_to_list_index;
};

}  // namespace typeart

#endif  // LLVM_MUST_SUPPORT_TYPECONFIG_H

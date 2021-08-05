// TypeART library
//
// Copyright (c) 2017-2021 TypeART Authors
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

#include <array>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace typeart {

class TypeDB final : public TypeDatabase {
 public:
  void clear();

  void registerStruct(const StructTypeInfo& struct_type) override;

  bool isUnknown(int id) const override;

  bool isValid(int id) const override;

  bool isReservedType(int id) const override;

  bool isBuiltinType(int id) const override;

  bool isStructType(int id) const override;

  bool isUserDefinedType(int id) const override;

  bool isVectorType(int id) const override;

  const std::string& getTypeName(int id) const override;

  const StructTypeInfo* getStructInfo(int id) const override;

  size_t getTypeSize(int id) const override;

  const std::vector<StructTypeInfo>& getStructList() const override;

  static const std::array<std::string, 11> BuiltinNames;
  static const std::array<size_t, 11> BuiltinSizes;
  static const std::string UnknownStructName;

 private:
  std::vector<StructTypeInfo> struct_info_vec;
  std::unordered_map<int, int> typeid_to_list_index;
};

}  // namespace typeart

#endif  // LLVM_MUST_SUPPORT_TYPECONFIG_H

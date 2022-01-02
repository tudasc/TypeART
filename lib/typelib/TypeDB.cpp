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

#include "TypeDB.h"

#include "TypeIO.h"
#include "support/Logger.h"
#include "typelib/TypeInterface.h"

#include <iostream>
#include <utility>

namespace typeart {

std::pair<std::unique_ptr<TypeDatabase>, std::error_code> make_database(const std::string& file) {
  auto type_db = std::make_unique<TypeDB>();
  auto loaded  = io::load(type_db.get(), file);
  if (!loaded) {
    LOG_DEBUG("Database file not found: " << file)
  }
  return {std::move(type_db), loaded.getError()};
}

const std::array<std::string, 11> TypeDB::BuiltinNames = {
    "int8", "int16", "int32", "int64", "half", "float", "double", "float128", "x86_float80", "ppc_float128", "pointer"};

// TODO: Builtin ID changes lead tsto wrong type sizes/names
const std::array<size_t, 11> TypeDB::BuiltinSizes = {1,  2,
                                                     4,  8,
                                                     2,  4,
                                                     8,  16,
                                                     16,  // TODO: Always correct?
                                                     16, sizeof(void*)};

// TypeInfo TypeDB::InvalidType = TypeInfo{BUILTIN, TA_UNKNOWN_TYPE};

const std::string TypeDB::UnknownStructName{"typeart_unknown_struct"};

void TypeDB::clear() {
  struct_info_vec.clear();
  typeid_to_list_index.clear();
  // reverseTypeMap.clear();
}

bool TypeDB::isBuiltinType(int type_id) const {
  return type_id >= TYPEART_INT8 && type_id < TYPEART_NUM_VALID_IDS;
}

bool TypeDB::isReservedType(int type_id) const {
  return type_id < TYPEART_NUM_RESERVED_IDS;
}

bool TypeDB::isStructType(int type_id) const {
  return type_id >= TYPEART_NUM_RESERVED_IDS;
}

bool TypeDB::isUserDefinedType(int type_id) const {
  const auto* structInfo = getStructInfo(type_id);
  return (structInfo != nullptr) &&
         ((static_cast<int>(structInfo->flag) & static_cast<int>(StructTypeFlag::USER_DEFINED)) != 0);
}

bool TypeDB::isVectorType(int type_id) const {
  const auto* structInfo = getStructInfo(type_id);
  return (structInfo != nullptr) &&
         ((static_cast<int>(structInfo->flag) & static_cast<int>(StructTypeFlag::LLVM_VECTOR)) != 0);
}

bool TypeDB::isValid(int type_id) const {
  if (isBuiltinType(type_id)) {
    return true;
  }
  return typeid_to_list_index.find(type_id) != typeid_to_list_index.end();
}

void TypeDB::registerStruct(const StructTypeInfo& struct_type) {
  if (isValid(struct_type.type_id) || !isStructType(struct_type.type_id)) {
    if (isBuiltinType(struct_type.type_id)) {
      LOG_ERROR("Built-in type ID used for struct " << struct_type.name);
    } else if (isReservedType(struct_type.type_id)) {
      LOG_ERROR("Type ID is reserved for builtin types. Struct: " << struct_type.name);
    } else if (isUnknown(struct_type.type_id)) {
      LOG_ERROR("Type ID is reserved for unknown types. Struct: " << struct_type.name);
    } else {
      LOG_ERROR("Struct type ID already registered for " << struct_type.name << ". Conflicting struct is "
                                                         << getStructInfo(struct_type.type_id)->name);
    }
    return;
  }
  struct_info_vec.push_back(struct_type);
  typeid_to_list_index.insert({struct_type.type_id, struct_info_vec.size() - 1});
}

const std::string& TypeDB::getTypeName(int type_id) const {
  if (isBuiltinType(type_id)) {
    return BuiltinNames[type_id];
  }
  if (isStructType(type_id)) {
    const auto* structInfo = getStructInfo(type_id);
    if (structInfo != nullptr) {
      return structInfo->name;
    }
  }
  return UnknownStructName;
}

size_t TypeDB::getTypeSize(int type_id) const {
  if (isReservedType(type_id)) {
    if (isBuiltinType(type_id)) {
      return BuiltinSizes[type_id];
    }
    return 0;
  }

  const auto* structInfo = getStructInfo(type_id);
  if (structInfo != nullptr) {
    return structInfo->extent;
  }
  return 0;
}

const StructTypeInfo* TypeDB::getStructInfo(int type_id) const {
  const auto index_iter = typeid_to_list_index.find(type_id);
  if (index_iter != typeid_to_list_index.end()) {
    return &struct_info_vec[index_iter->second];
  }
  return nullptr;
}

const std::vector<StructTypeInfo>& TypeDB::getStructList() const {
  return struct_info_vec;
}

bool TypeDB::isUnknown(int type_id) const {
  return type_id == TYPEART_UNKNOWN_TYPE;
}

}  // namespace typeart

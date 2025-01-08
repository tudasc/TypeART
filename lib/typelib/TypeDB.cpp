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

#include "TypeDB.h"

#include "TypeIO.h"
#include "support/Logger.h"
#include "typelib/TypeInterface.h"

#include <ccomplex>
#include <cstddef>
#include <iostream>
#include <utility>

namespace typeart {

namespace builtins {

// namespace {
// #pragma GCC diagnostic ignored "-Wc99-extensions"
// #pragma GCC diagnostic push
// inline constexpr auto size_complex_8  = sizeof(float _Complex);
// inline constexpr auto size_complex_16 = sizeof(double _Complex);
// inline constexpr auto size_complex_32 = sizeof(long double _Complex);
// #pragma GCC diagnostic pop
// }  // namespace

#define FOR_EACH_TYPEART_BUILTIN(X)                          \
  X(TYPEART_UNKNOWN_TYPE, "typeart_unknown_type", 0)         \
  X(TYPEART_POINTER, "ptr", sizeof(void*))                   \
  X(TYPEART_VOID, "void*", sizeof(void*))                    \
  X(TYPEART_NULLPOINTER, "nullptr_t", sizeof(void*))         \
  X(TYPEART_BOOL, "bool", sizeof(bool))                      \
  X(TYPEART_CHAR_8, "char", sizeof(char))                    \
  X(TYPEART_UCHAR_8, "unsigned char", sizeof(unsigned char)) \
  X(TYPEART_UTF_CHAR_8, "char8_t", sizeof(char))             \
  X(TYPEART_UTF_CHAR_16, "char16_t", sizeof(char16_t))       \
  X(TYPEART_UTF_CHAR_32, "char32_t", sizeof(char32_t))       \
  X(TYPEART_INT_8, "int8_t", sizeof(int8_t))                 \
  X(TYPEART_INT_16, "short", sizeof(int16_t))                \
  X(TYPEART_INT_32, "int", sizeof(int32_t))                  \
  X(TYPEART_INT_64, "long int", sizeof(int64_t))             \
  X(TYPEART_INT_128, "int128_t", 16)                         \
  X(TYPEART_UINT_8, "uint8_t", sizeof(uint8_t))              \
  X(TYPEART_UINT_16, "unsigned short", sizeof(uint16_t))     \
  X(TYPEART_UINT_32, "unsigned int", sizeof(uint32_t))       \
  X(TYPEART_UINT_64, "unsigned long int", sizeof(uint64_t))  \
  X(TYPEART_UINT_128, "uint128_t", 16)                       \
  X(TYPEART_FLOAT_8, "float8_t", 1)                          \
  X(TYPEART_FLOAT_16, "float16_t", 2)                        \
  X(TYPEART_FLOAT_32, "float", sizeof(float))                \
  X(TYPEART_FLOAT_64, "double", sizeof(double))              \
  X(TYPEART_FLOAT_128, "long double", sizeof(long double))   \
  X(TYPEART_COMPLEX_8, "float complex", 1)                   \
  X(TYPEART_COMPLEX_16, "double complex", 2)                 \
  X(TYPEART_COMPLEX_32, "long double complex", 4)

#define TYPENAME(enum_name, str_name, size) std::string{str_name},
#define SIZE(enum_name, str_name, size)     (size),
BuiltInQuery::BuiltInQuery() : names{FOR_EACH_TYPEART_BUILTIN(TYPENAME)}, sizes{FOR_EACH_TYPEART_BUILTIN(SIZE)} {
}
#undef SIZE
#undef TYPENAME

}  // namespace builtins

std::pair<std::unique_ptr<TypeDatabase>, std::error_code> make_database(const std::string& file) {
  auto type_db = std::make_unique<TypeDB>();
  auto loaded  = io::load(type_db.get(), file);
  if (!loaded) {
    LOG_DEBUG("Database file not found: " << file)
  }
  return {std::move(type_db), loaded.getError()};
}

const std::string UnknownStructName{"typeart_unknown_struct"};

using namespace builtins;

void TypeDB::clear() {
  struct_info_vec.clear();
  typeid_to_list_index.clear();
  // reverseTypeMap.clear();
}

bool TypeDB::isBuiltinType(int type_id) const {
  return BuiltInQuery::is_builtin_type(type_id);
}

bool TypeDB::isReservedType(int type_id) const {
  return BuiltInQuery::is_reserved_type(type_id);
}

bool TypeDB::isStructType(int type_id) const {
  return BuiltInQuery::is_userdef_type(type_id);
}

bool TypeDB::isUnknown(int type_id) const {
  return BuiltInQuery::is_unknown_type(type_id);
}

bool TypeDB::isUserDefinedType(int type_id) const {
  const auto* structInfo = getStructInfo(type_id);
  LOG_DEBUG(structInfo->name << " " << static_cast<int>(structInfo->flag) << " "
                             << (static_cast<int>(structInfo->flag) == static_cast<int>(StructTypeFlag::USER_DEFINED)))
  return (structInfo != nullptr) &&
         (static_cast<int>(structInfo->flag) == static_cast<int>(StructTypeFlag::USER_DEFINED));
}

bool TypeDB::isVectorType(int type_id) const {
  const auto* structInfo = getStructInfo(type_id);
  LOG_DEBUG(structInfo->name << " " << static_cast<int>(structInfo->flag) << " "
                             << (static_cast<int>(structInfo->flag) == static_cast<int>(StructTypeFlag::LLVM_VECTOR)))
  return (structInfo != nullptr) &&
         (static_cast<int>(structInfo->flag) == static_cast<int>(StructTypeFlag::LLVM_VECTOR));
}

bool TypeDB::isValid(int type_id) const {
  if (isBuiltinType(type_id)) {
    return true;
  }
  return typeid_to_list_index.find(type_id) != typeid_to_list_index.end();
}

void TypeDB::registerStruct(const StructTypeInfo& struct_type, bool overwrite) {
  if (isValid(struct_type.type_id) || !isStructType(struct_type.type_id)) {
    if (isBuiltinType(struct_type.type_id)) {
      LOG_ERROR("Built-in type ID used for struct " << struct_type.name);
    } else if (isReservedType(struct_type.type_id)) {
      LOG_ERROR("Type ID is reserved for builtin types. Struct: " << struct_type.name);
    } else if (isUnknown(struct_type.type_id)) {
      LOG_ERROR("Type ID is reserved for unknown types. Struct: " << struct_type.name);
    } else {
      if (!overwrite) {
        LOG_ERROR("Struct type ID already registered for " << struct_type.name << ". Conflicting struct is "
                                                           << getStructInfo(struct_type.type_id)->name);
        return;
      }
      LOG_DEBUG("Overwrite struct " << struct_type.name)
      auto& info = *getStructInfo(struct_type.type_id);
      info       = struct_type;
    }
    return;
  }
  struct_info_vec.push_back(struct_type);
  typeid_to_list_index.insert({struct_type.type_id, struct_info_vec.size() - 1});
}

const std::string& TypeDB::getTypeName(int type_id) const {
  if (isBuiltinType(type_id)) {
    return builtins.get_name(type_id);
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
      return builtins.get_size(type_id);
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

StructTypeInfo* TypeDB::getStructInfo(int type_id) {
  const auto index_iter = typeid_to_list_index.find(type_id);
  if (index_iter != typeid_to_list_index.end()) {
    return &struct_info_vec[index_iter->second];
  }
  return nullptr;
}

const std::vector<StructTypeInfo>& TypeDB::getStructList() const {
  return struct_info_vec;
}

}  // namespace typeart

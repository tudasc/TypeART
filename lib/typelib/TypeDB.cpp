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

namespace {
inline constexpr auto size_complex_8  = sizeof(float _Complex);
inline constexpr auto size_complex_16 = sizeof(double _Complex);
inline constexpr auto size_complex_32 = sizeof(long double _Complex);
}  // namespace

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
  X(TYPEART_COMPLEX_8, "float complex", size_complex_8)      \
  X(TYPEART_COMPLEX_16, "double complex", size_complex_16)   \
  X(TYPEART_COMPLEX_32, "long double complex", size_complex_32)

std::pair<std::unique_ptr<TypeDatabase>, std::error_code> make_database(const std::string& file) {
  auto type_db = std::make_unique<TypeDB>();
  auto loaded  = io::load(type_db.get(), file);
  if (!loaded) {
    LOG_DEBUG("Database file not found: " << file)
  }
  return {std::move(type_db), loaded.getError()};
}

struct BuiltInHandler {
 private:
#define TYPENAME(enum_name, str_name, size) str_name,
  inline static const std::array<std::string, TYPEART_NUM_VALID_IDS> names{FOR_EACH_TYPEART_BUILTIN(TYPENAME)};
#undef TYPENAME

#define SIZE(enum_name, str_name, size) (size),
  inline static constexpr std::array<size_t, TYPEART_NUM_VALID_IDS> sizes{FOR_EACH_TYPEART_BUILTIN(SIZE)};
#undef SIZE

  enum AccessIdx { NAME = 0, SIZE = 1 };

  template <typename T, AccessIdx pos = AccessIdx::NAME>
  constexpr static T type_info(int type_id) {
    if constexpr (pos == AccessIdx::NAME) {
      if (type_id < 0 || type_id >= TYPEART_NUM_VALID_IDS) {
        return names[TYPEART_UNKNOWN_TYPE];
      }
      return names[type_id];
    } else {
      if (type_id < 0 || type_id >= TYPEART_NUM_VALID_IDS) {
        return sizes[TYPEART_UNKNOWN_TYPE];
      }
      return sizes[type_id];
    }
  }

 public:
  static constexpr size_t get_size(int type_id) {
    return type_info<size_t, AccessIdx::SIZE>(type_id);
  }

  static const std::string& get_name(int type_id) {
    return type_info<const std::string&, AccessIdx::NAME>(type_id);
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

const std::string UnknownStructName{"typeart_unknown_struct"};

void TypeDB::clear() {
  struct_info_vec.clear();
  typeid_to_list_index.clear();
  // reverseTypeMap.clear();
}

bool TypeDB::isBuiltinType(int type_id) const {
  return BuiltInHandler::is_builtin_type(type_id);
}

bool TypeDB::isReservedType(int type_id) const {
  return BuiltInHandler::is_reserved_type(type_id);
}

bool TypeDB::isStructType(int type_id) const {
  return BuiltInHandler::is_userdef_type(type_id);
}

bool TypeDB::isUnknown(int type_id) const {
  return BuiltInHandler::is_unknown_type(type_id);
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
    return BuiltInHandler::get_name(type_id);
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
      return BuiltInHandler::get_size(type_id);
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

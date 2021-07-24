//
// Created by sebastian on 22.03.18.
//

#include "TypeDB.h"

#include "TypeIO.h"
#include "support/Logger.h"
#include "typelib/TypeInterface.h"

#include <iostream>
#include <utility>

namespace typeart {

std::pair<std::unique_ptr<TypeDatabase>, std::error_code> make_database(const std::string& file) {
  auto db     = std::make_unique<TypeDB>();
  auto loaded = io::load(db.get(), file);
  if (!loaded) {
    LOG_DEBUG("Database file not found: " << file)
  }
  return {std::move(db), loaded.getError()};
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

const std::string TypeDB::UnknownStructName{"UnknownStruct"};

void TypeDB::clear() {
  structInfoList.clear();
  id2Idx.clear();
  // reverseTypeMap.clear();
}

bool TypeDB::isBuiltinType(int id) const {
  return id >= TA_INT8 && id < TA_NUM_VALID_IDS;
}

bool TypeDB::isReservedType(int id) const {
  return id < TA_NUM_RESERVED_IDS;
}

bool TypeDB::isStructType(int id) const {
  return id >= TA_NUM_RESERVED_IDS;
}

bool TypeDB::isUserDefinedType(int id) const {
  auto structInfo = getStructInfo(id);
  return (structInfo != nullptr) &&
         ((static_cast<int>(structInfo->flag) & static_cast<int>(StructTypeFlag::USER_DEFINED)) != 0);
}

bool TypeDB::isVectorType(int id) const {
  auto structInfo = getStructInfo(id);
  return (structInfo != nullptr) &&
         ((static_cast<int>(structInfo->flag) & static_cast<int>(StructTypeFlag::LLVM_VECTOR)) != 0);
}

bool TypeDB::isValid(int id) const {
  if (isBuiltinType(id)) {
    return true;
  }
  return id2Idx.find(id) != id2Idx.end();
}

void TypeDB::registerStruct(const StructTypeInfo& structType) {
  if (isValid(structType.id)) {
    LOG_ERROR("Invalid type ID for struct " << structType.name);
    if (isReservedType(structType.id)) {
      LOG_ERROR("Type ID is reserved for builtin types");
    } else {
      LOG_ERROR("Conflicting struct is " << getStructInfo(structType.id)->name);
    }
    // TODO: Error handling
    return;
  }
  structInfoList.push_back(structType);
  id2Idx.insert({structType.id, structInfoList.size() - 1});
  // reverseTypeMap.insert({id, typeName});
}

const std::string& TypeDB::getTypeName(int id) const {
  if (isBuiltinType(id)) {
    return BuiltinNames[id];
  }
  if (isStructType(id)) {
    const auto* structInfo = getStructInfo(id);
    if (structInfo != nullptr) {
      return structInfo->name;
    }
  }
  return UnknownStructName;
}

size_t TypeDB::getTypeSize(int id) const {
  if (isReservedType(id)) {
    if (isBuiltinType(id)) {
      return BuiltinSizes[id];
    }
    return 0;
  }

  const auto* structInfo = getStructInfo(id);
  if (structInfo != nullptr) {
    return structInfo->extent;
  }
  return 0;
}

const StructTypeInfo* TypeDB::getStructInfo(int id) const {
  auto it = id2Idx.find(id);
  if (it != id2Idx.end()) {
    return &structInfoList[it->second];
  }
  return nullptr;
}

const std::vector<StructTypeInfo>& TypeDB::getStructList() const {
  return structInfoList;
}

bool TypeDB::isUnknown(int id) const {
  return id == TA_UNKNOWN_TYPE;
}

}  // namespace typeart

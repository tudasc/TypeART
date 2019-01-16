//
// Created by sebastian on 22.03.18.
//

#include "TypeDB.h"

//#include <form.h> FIXME why needed?
#include <iostream>

namespace typeart {

std::string TypeDB::BuiltinNames[] = {"int8",   "int16",    "int32",       "int64",        "half",   "float",
                                      "double", "float128", "x86_float80", "ppc_float128", "pointer"};

// TODO: Builtin ID changes lead to wrong type sizes/names
size_t TypeDB::BuiltinSizes[] = {1,  2,
                                 4,  8,
                                 2,  4,
                                 8,  16,
                                 16,  // TODO: Always correct?
                                 16, sizeof(void*)};

// TypeInfo TypeDB::InvalidType = TypeInfo{BUILTIN, TA_UNKNOWN_TYPE};

std::string TypeDB::UnknownStructName{"UnknownStruct"};

TypeDB::TypeDB() = default;

void TypeDB::clear() {
  structInfoList.clear();
  id2Idx.clear();
  // reverseTypeMap.clear();
}

bool TypeDB::isBuiltinType(int id) const {
  return id < TA_NUM_VALID_IDS;
}

bool TypeDB::isReservedType(int id) const {
  return id < TA_NUM_RESERVED_IDS;
}

bool TypeDB::isStructType(int id) const {
  return id >= TA_NUM_RESERVED_IDS;
}

bool TypeDB::isUserDefinedType(int id) const {
  auto structInfo = getStructInfo(id);
  return structInfo && (structInfo->flags & static_cast<int>(TA_USER_DEF));
}

bool TypeDB::isVectorType(int id) const {
  auto structInfo = getStructInfo(id);
  return structInfo && (structInfo->flags & static_cast<int>(TA_VEC));
}

bool TypeDB::isValid(int id) const {
  if (isBuiltinType(id)) {
    return true;
  }
  return id2Idx.find(id) != id2Idx.end();
}

void TypeDB::registerStruct(StructTypeInfo structType) {
  if (isValid(structType.id)) {
    std::cerr << "Invalid type ID for struct " << structType.name << std::endl;
    if (isReservedType(structType.id)) {
      std::cerr << "Type ID is reserved for builtin types" << std::endl;
    } else {
      std::cerr << "Conflicting struct is " << getStructInfo(structType.id)->name << std::endl;
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
    if (structInfo) {
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
  if (structInfo) {
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

}  // namespace typeart

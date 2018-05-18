//
// Created by sebastian on 22.03.18.
//

#include "TypeConfig.h"
#include <form.h>
#include <iostream>

namespace must {

std::string TypeConfig::builtinNames[] = {"char", "uchar", "short", "ushort", "int", "uint", "long", "ulong", "float", "double", "invalid"};

TypeInfo TypeConfig::InvalidType = TypeInfo{BUILTIN, INVALID};

TypeConfig::TypeConfig() {
}

void TypeConfig::clear() {
  structInfoList.clear();
  id2Idx.clear();
  // reverseTypeMap.clear();
}

bool TypeConfig::isBuiltinType(int id) const {
  return id < N_BUILTIN_TYPES;
}



bool TypeConfig::isStructType(int id) const {
  return id2Idx.find(id) != id2Idx.end();
}

bool TypeConfig::hasTypeID(int id) const {
  if (isBuiltinType(id)) {
    return true;
  }
  return id2Idx.find(id) != id2Idx.end();
}

void TypeConfig::registerStruct(StructTypeInfo structType) {
  if (hasTypeID(structType.id)) {
    std::cerr << "Invalid type ID for struct " << structType.name << std::endl;
    if (isBuiltinType(structType.id)) {
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

std::string TypeConfig::getTypeName(int id) const {
  if (isBuiltinType(id)) {
    return builtinNames[id];
  }
  if (isStructType(id)) {
    const auto* structInfo = getStructInfo(id);
    if (structInfo) {
      return structInfo->name;
    }
  }
  return "UnknownStruct";
}

// std::vector<std::string> TypeConfig::getTypeList() const {
//  std::vector<std::string> typeIDs;
//  typeIDs.reserve(typeMap.size());
//  for (const auto& entry : typeMap) {
//    typeIDs.push_back(entry.first);
//  }
//  return typeIDs;
//}

int TypeConfig::getBuiltinTypeSize(int id) const {
  switch (id) {
    case C_CHAR:
    case C_UCHAR:
      return 1;
    case C_SHORT:
    case C_USHORT:
      return 2;
    case C_INT:
    case C_FLOAT:
    case C_UINT:
      return 4;
    case C_LONG:
    case C_DOUBLE:
    case C_ULONG:
      return 8;
    default:
      return -1;
  }
}

const StructTypeInfo* TypeConfig::getStructInfo(int id) const {
  auto it = id2Idx.find(id);
  if (it != id2Idx.end()) {
    return &structInfoList[it->second];
  }
  return nullptr;
}

TypeInfo TypeConfig::getTypeInfo(int id) const {
  if (isBuiltinType(id)) {
    return TypeInfo{BUILTIN, id};
  }
  if (isStructType(id)) {
    return TypeInfo{STRUCT, id};
  }
  return InvalidType;
}

const std::vector<StructTypeInfo>& TypeConfig::getStructList() const {
  return structInfoList;
}
}
//
// Created by sebastian on 22.03.18.
//

#include "TypeConfig.h"
#include <form.h>
#include <iostream>

namespace must {

std::string TypeConfig::builtinNames[] = {"int", "uint", "char", "uchar", "long", "ulong", "float", "double"};

TypeConfig::TypeConfig() {
}

void TypeConfig::clear() {
  structMap.clear();
  // reverseTypeMap.clear();
}

bool TypeConfig::isBuiltinType(int id) const {
  return id < N_BUILTIN_TYPES;
}

bool TypeConfig::isStructType(int id) const {
  return structMap.find(id) != structMap.end();
}

bool TypeConfig::hasTypeID(int id) const {
  if (isBuiltinType(id)) {
    return true;
  }
  return structMap.find(id) != structMap.end();
}

void TypeConfig::registerStruct(StructTypeInfo structType) {
  if (hasTypeID(structType.id)) {
    std::cerr << "Invalid type ID for struct " << structType.name << std::endl;
    // TODO: Error handling
    return;
  }
  structMap.insert({structType.id, structType});
  // reverseTypeMap.insert({id, typeName});
}

std::string TypeConfig::getTypeName(int id) const {
  if (isBuiltinType(id)) {
    return builtinNames[id];
  }
  if (isStructType(id)) {
    auto structInfo = getStructInfo(id);
    return structInfo.name;
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

StructTypeInfo TypeConfig::getStructInfo(int id) const {
  return structMap.find(id)->second;
}

TypeInfo TypeConfig::getTypeInfo(int id) const {
  if (isBuiltinType(id)) {
    return TypeInfo{BUILTIN, id};
  }
  if (isStructType(id)) {
    return TypeInfo{STRUCT, id};
  }
}

std::vector<StructTypeInfo> TypeConfig::getStructList() const {
  std::vector<StructTypeInfo> structTypes;
  structTypes.reserve(structMap.size());
  for (const auto& entry : structMap) {
    structTypes.push_back(entry.second);
  }
  return structTypes;
}
}
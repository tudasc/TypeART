//
// Created by sebastian on 22.03.18.
//

#include "TypeConfig.h"

namespace must {

TypeConfig::TypeConfig() {
}

void TypeConfig::clear() {
  typeMap.clear();
  reverseTypeMap.clear();
}

void TypeConfig::registerType(std::string typeName, int id) {
  typeMap.insert({typeName, id});
  // TODO: There's probably a better way to do this...
  reverseTypeMap.insert({id, typeName});
}

int TypeConfig::getTypeID(std::string typeName) const {
  auto it = typeMap.find(typeName);
  if (it != typeMap.end()) {
    return it->second;
  }
  return UNDEFINED;
}

bool TypeConfig::hasTypeID(std::string typeName) const {
  return typeMap.find(typeName) != typeMap.end();
}

std::string TypeConfig::getTypeName(int id) const {
  auto it = reverseTypeMap.find(id);
  if (it != reverseTypeMap.end()) {
    return it->second;
  }
  return "Unknown";
}

std::vector<std::string> TypeConfig::getTypeList() const {
  std::vector<std::string> typeIDs;
  typeIDs.reserve(typeMap.size());
  for (const auto& entry : typeMap) {
    typeIDs.push_back(entry.first);
  }
  return typeIDs;
}
}
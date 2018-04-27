//
// Created by sebastian on 22.03.18.
//

#include "ConfigIO.h"

#include <fstream>

namespace must {

ConfigIO::ConfigIO(TypeConfig* config) : config(config) {
}

bool ConfigIO::load(std::string file) {
  config->clear();
  std::ifstream is;
  is.open(file);
  if (!is.is_open()) {
    return false;
  }
  std::string name;
  int id;
  /*while (is >> name >> id) {
    config->registerType(name, id);
  }*/ // TODO
  is.close();
  return true;
}

bool ConfigIO::store(std::string file) const {
  std::ofstream os;
  os.open(file);
  if (!os.is_open()) {
    return false;
  }
  for (const auto& structInfo: config->getStructList()) {
    os << serialize(structInfo) << std::endl;
    //os << typeName << " " << id << std::endl;
  }
  os.close();
  return true;
}

std::string ConfigIO::serialize(StructTypeInfo structInfo) const
{
  // TODO
  return "";
}

StructTypeInfo ConfigIO::deserialize(std::string infoString) const
{
  // TODO
  return StructTypeInfo();
}


}
//
// Created by sebastian on 22.03.18.
//

#include "TypeIO.h"

#include <algorithm>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <sstream>

namespace must {

TypeIO::TypeIO(TypeDB* typeDB) : typeDB(typeDB) {
}

bool TypeIO::load(std::string file) {
  typeDB->clear();
  std::ifstream is;
  is.open(file);
  if (!is.is_open()) {
    return false;
  }

  std::string line;
  while (std::getline(is, line)) {
    // TODO: Error handling
    if (isComment(line)) {
      continue;
    }
    StructTypeInfo structInfo = deserialize(line);
    // std::cout << "Struct: " << structInfo.name << ", " << structInfo.id << std::endl;
    typeDB->registerStruct(structInfo);
  }
  is.close();
  return true;
}

bool TypeIO::store(std::string file) const {
  std::ofstream os;
  os.open(file);
  if (!os.is_open()) {
    return false;
  }
  os << "# MUST Type Support Mapping" << std::endl;
  os << "#" << std::endl;
  os << "# ID"
     << "\t"
     << "Name"
     << "\t"
     << "NumBytes"
     << "\t"
     << "NumMembers"
     << "\t"
     << "(Offset, TypeKind, TypeID, ArraySize)*" << std::endl;
  os << "# --------" << std::endl;
  for (const auto& structInfo : typeDB->getStructList()) {
    os << serialize(structInfo) << std::endl;
    // os << typeName << " " << id << std::endl;
  }
  os.close();
  return true;
}

std::string TypeIO::serialize(StructTypeInfo structInfo) const {
  std::stringstream ss;
  ss << structInfo.id << "\t" << structInfo.name << "\t" << structInfo.extent << "\t" << structInfo.numMembers << "\t";
  assert(structInfo.numMembers == structInfo.offsets.size() && structInfo.numMembers == structInfo.memberTypes.size() &&
         structInfo.numMembers == structInfo.arraySizes.size() && "Invalid vector sizes in struct info");
  for (int i = 0; i < structInfo.numMembers; i++) {
    ss << structInfo.offsets[i] << "," << structInfo.memberTypes[i].kind << "," << structInfo.memberTypes[i].id << ","
       << structInfo.arraySizes[i] << "\t";
  }
  return ss.str();
}

StructTypeInfo TypeIO::deserialize(std::string infoString) const {
  auto split = [](const char* str, char c = ' ') -> std::vector<std::string> {
    std::vector<std::string> result;
    do {
      const char* begin = str;
      while (*str != c && *str)
        str++;
      result.push_back(std::string(begin, str));
    } while (0 != *str++);
    return result;
  };

  int id = INVALID;
  std::string name;
  int numBytes = 0;
  int numMembers = 0;
  std::vector<int> offsets;
  std::vector<TypeInfo> memberTypes;
  std::vector<int> arraySizes;

  auto entries = split(infoString.c_str(), '\t');
  id = std::stoi(entries[0]);
  name = entries[1];
  numBytes = std::stoi(entries[2]);
  numMembers = std::stoi(entries[3]);

  // iss >> id >> name >> numBytes >> numMembers;

  // std::cerr << "Struct deserialized: " << id << ", " << name << ", " << numBytes << ", " << numMembers << std::endl;

  const int memberStartIndex = 4;

  std::string memberInfoString;
  for (int i = 0; i < numMembers; i++) {
    memberInfoString = entries[memberStartIndex + i];
    auto memberInfoTokens = split(memberInfoString.c_str(), ',');
    if (memberInfoTokens.size() == 4) {
      offsets.push_back(std::stoi(memberInfoTokens[0]));
      memberTypes.push_back({TypeKind(std::stoi(memberInfoTokens[1])), std::stoi({memberInfoTokens[2]})});
      arraySizes.push_back(std::stoi(memberInfoTokens[3]));
    } else {
      // TODO: Handle error
      std::cerr << "Invalid struct member description string: " << memberInfoString << std::endl;
    }
  }

  // TODO: This should not be an assertion
  assert(numMembers == offsets.size() && numMembers == memberTypes.size() && numMembers == arraySizes.size() &&
         "Invalid vector sizes in struct info");

  return StructTypeInfo{id, name, numBytes, numMembers, offsets, memberTypes, arraySizes};
}

bool TypeIO::isComment(std::string line) const {
  line.erase(line.begin(), std::find_if(line.begin(), line.end(), [](int ch) { return !std::isspace(ch); }));
  return line.empty() || line.front() == '#';
}
}
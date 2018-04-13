//
// Created by sebastian on 22.03.18.
//

#ifndef LLVM_MUST_SUPPORT_TYPECONFIG_H
#define LLVM_MUST_SUPPORT_TYPECONFIG_H

#include <map>
#include <vector>

namespace must {

class TypeConfig {
 public:
  TypeConfig();

  void clear();

  void registerType(std::string typeName, int id);

  int getTypeID(std::string typeName) const;

  bool hasTypeID(std::string typeName) const;

  std::string getTypeName(int id) const;

  std::vector<std::string> getTypeList() const;

  static const int UNDEFINED = -1;

 private:
  std::map<std::string, int> typeMap;
  std::map<int, std::string> reverseTypeMap;
};
}

#endif  // LLVM_MUST_SUPPORT_TYPECONFIG_H

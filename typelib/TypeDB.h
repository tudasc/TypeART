//
// Created by sebastian on 22.03.18.
//

#ifndef LLVM_MUST_SUPPORT_TYPECONFIG_H
#define LLVM_MUST_SUPPORT_TYPECONFIG_H

#include "TypeInterface.h"
#include <map>
#include <vector>

namespace must {

using TypeInfo = must_type_info;
using BuiltinType = must_builtin_type;
using TypeKind = must_type_kind;

struct StructTypeInfo {
  int id;
  std::string name;
  int extent;
  int numMembers;
  std::vector<int> offsets;
  std::vector<TypeInfo> memberTypes;
  std::vector<int> arraySizes;
};

class TypeDB {
 public:
  TypeDB();

  void clear();

  void registerStruct(StructTypeInfo structInfo);

  bool isValid(int id) const;

  bool isBuiltinType(int id) const;

  bool isStructType(int id) const;

  const std::string& getTypeName(int id) const;

  const StructTypeInfo* getStructInfo(int id) const;

  TypeInfo getTypeInfo(int id) const;

  int getBuiltinTypeSize(int id) const;

  const std::vector<StructTypeInfo>& getStructList() const;

  static std::string builtinNames[];

  static TypeInfo InvalidType;

 private:
  std::vector<StructTypeInfo> structInfoList;
  std::map<int, int> id2Idx;
};
}

#endif  // LLVM_MUST_SUPPORT_TYPECONFIG_H
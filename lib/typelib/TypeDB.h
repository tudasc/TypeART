//
// Created by sebastian on 22.03.18.
//

#ifndef LLVM_MUST_SUPPORT_TYPECONFIG_H
#define LLVM_MUST_SUPPORT_TYPECONFIG_H

#include "TypeInterface.h"

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

namespace typeart {

using BuiltinType = typeart_builtin_type;

struct StructTypeInfo {
  int id;
  std::string name;
  size_t extent;
  size_t numMembers;
  std::vector<size_t> offsets;
  std::vector<int> memberTypes;
  std::vector<size_t> arraySizes;
  int flags;
};

class TypeDB {
 public:
  TypeDB();

  void clear();

  void registerStruct(const StructTypeInfo& structInfo);

  bool isValid(int id) const;

  bool isReservedType(int id) const;

  bool isBuiltinType(int id) const;

  bool isStructType(int id) const;

  bool isUserDefinedType(int id) const;

  bool isVectorType(int id) const;

  const std::string& getTypeName(int id) const;

  const StructTypeInfo* getStructInfo(int id) const;

  size_t getTypeSize(int id) const;

  const std::vector<StructTypeInfo>& getStructList() const;

  static const std::array<std::string, 11> BuiltinNames;
  static const std::array<size_t, 11> BuiltinSizes;
  static const std::string UnknownStructName;

 private:
  std::vector<StructTypeInfo> structInfoList;
  std::unordered_map<int, int> id2Idx;
};

}  // namespace typeart

#endif  // LLVM_MUST_SUPPORT_TYPECONFIG_H

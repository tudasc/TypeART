//
// Created by sebastian on 22.03.18.
//

#ifndef LLVM_MUST_SUPPORT_TYPECONFIG_H
#define LLVM_MUST_SUPPORT_TYPECONFIG_H

#include <map>
#include <vector>

namespace must {

enum BuiltinTypes {
    C_INT,
    C_UINT,
    C_CHAR,
    C_UCHAR,
    C_LONG,
    C_ULONG,
    C_FLOAT,
    C_DOUBLE,
    INVALID,
    N_BUILTIN_TYPES
};

enum TypeKind {
    BUILTIN,
    STRUCT,
    POINTER
};

struct TypeInfo {
    TypeKind kind;
    int id;
};

struct StructTypeInfo {
    int id;
    int numMembers;
    std::vector<TypeInfo> memberTypes;
    std::vector<int> arraySizes;
    std::vector<int> offsets;
    int numBytes;
    std::string name;
};

class TypeConfig {
 public:
  TypeConfig();

  void clear();

  void registerStruct(StructTypeInfo structInfo);

  bool hasTypeID(int id) const;

  bool isBuiltinType(int id) const;

  bool isStructType(int id) const;

  //int getTypeID(std::string typeName) const;

  //bool hasTypeID(std::string typeName) const;

  std::string getTypeName(int id) const;

  StructTypeInfo getStructInfo(int id) const;

  TypeInfo getTypeInfo(int id) const;

  std::vector<StructTypeInfo> getStructList() const;

  static std::string builtinNames[];

 private:
  std::map<int, StructTypeInfo> structMap;
};
}

#endif  // LLVM_MUST_SUPPORT_TYPECONFIG_H

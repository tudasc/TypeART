//
// Created by sebastian on 27.04.18.
//

#include "TypeManager.h"
#include "support/TypeUtil.h"
#include <TypeIO.h>
#include <iostream>

namespace must {

namespace tu = util::type;
using namespace llvm;

TypeManager::TypeManager() : structCount(0) {
}

bool TypeManager::load(std::string file) {
  TypeIO cio(&typeDB);
  if (cio.load(file)) {
    structMap.clear();
    for (auto& structInfo : typeDB.getStructList()) {
      structMap.insert({structInfo.name, structInfo.id});
    }
    structCount = structMap.size();
    return true;
  }
  return false;
}

bool TypeManager::store(std::string file) {
  TypeIO cio(&typeDB);
  return cio.store(file);
}

int TypeManager::getOrRegisterType(llvm::Type* type, const llvm::DataLayout& dl) {
  auto& c = type->getContext();
  switch (type->getTypeID()) {
    case llvm::Type::IntegerTyID:

      if (type == Type::getInt8Ty(c)) {
        return C_CHAR;
      } else if (type == Type::getInt32Ty(c)) {
        return C_INT;
      } else if (type == Type::getInt64Ty(c)) {
        return C_LONG;
      } else {
        return INVALID;
      }
    // TODO: Unsigned types are not explicitly represented in LLVM. How to handle?
    case llvm::Type::FloatTyID:
      return C_FLOAT;
    case llvm::Type::DoubleTyID:
      return C_DOUBLE;
    case llvm::Type::StructTyID:
      return getOrRegisterStruct(dyn_cast<StructType>(type), dl);
    default:
      break;
  }
  return INVALID;
}

int TypeManager::getOrRegisterStruct(llvm::StructType* type, const llvm::DataLayout& dl) {
  auto name = type->getStructName();
  // std::cerr << "Looking up struct " << name.str() << std::endl;
  auto it = structMap.find(name);
  if (it != structMap.end()) {
    // std::cerr << "Found!" << std::endl;
    return it->second;
  }

  // std::cerr << "Registered structs: " << std::endl;
  // for (auto info : typeDB.getStructList()) {
  //  std::cerr << info.name <<", " << info.id << std::endl;
  //}

  // Get next ID and register struct
  int id = N_BUILTIN_TYPES + structCount;
  structCount++;

  int n = type->getStructNumElements();

  std::vector<int> offsets;
  std::vector<int> arraySizes;
  std::vector<TypeInfo> memberTypeInfo;

  const StructLayout* layout = dl.getStructLayout(type);

  for (int i = 0; i < n; i++) {
    auto memberType = type->getStructElementType(i);
    int memberID = INVALID;
    TypeKind kind;

    int arraySize = 1;

    if (memberType->isArrayTy()) {
      arraySize = memberType->getArrayNumElements();
      memberType = memberType->getArrayElementType();
    }

    if (memberType->isStructTy()) {
      kind = STRUCT;
      if (memberType->getStructName() == name) {
        memberID = id;
      } else {
        // TODO: Infinite cycle possible?
        memberID = getOrRegisterType(memberType, dl);
      }
    } else if (memberType->isPointerTy()) {
      kind = POINTER;
      // TODO: Do we need a type ID for pointers?
    } else if (memberType->isSingleValueType()) {
      kind = BUILTIN;
      memberID = getOrRegisterType(memberType, dl);
    } else {
      // TODO: Any other types?
      memberType->dump();
      assert(false && "Encountered unhandled type");
    }

    int offset = layout->getElementOffset(i);
    offsets.push_back(offset);
    arraySizes.push_back(arraySize);
    memberTypeInfo.push_back({kind, memberID});
  }

  int numBytes = layout->getSizeInBytes();

  StructTypeInfo structInfo{id, name, numBytes, n, offsets, memberTypeInfo, arraySizes};
  typeDB.registerStruct(structInfo);

  structMap.insert({name, id});
  return id;
}
}
//
// Created by sebastian on 27.04.18.
//

#include "TypeManager.h"
#include "TypeIO.h"
#include "support/Logger.h"
#include "support/TypeUtil.h"
#include "support/Util.h"

#include <iostream>

namespace tu = util::type;

namespace typeart {

using namespace llvm;

TypeManager::TypeManager(std::string file) : file(file), structCount(0) {
}

bool TypeManager::load() {
  TypeIO cio(typeDB);
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

bool TypeManager::store() {
  TypeIO cio(typeDB);
  return cio.store(file);
}

int TypeManager::getOrRegisterType(llvm::Type* type, const llvm::DataLayout& dl) {
  auto& c = type->getContext();
  switch (type->getTypeID()) {
    case llvm::Type::IntegerTyID:

      if (type == Type::getInt8Ty(c)) {
        return C_CHAR;
      } else if (type == Type::getInt16Ty(c)) {
        return C_SHORT;
      } else if (type == Type::getInt32Ty(c)) {
        return C_INT;
      } else if (type == Type::getInt64Ty(c)) {
        return C_LONG;
      } else {
        return UNKNOWN;
      }
    // TODO: Unsigned types are not supported as of now
    case llvm::Type::FloatTyID:
      return C_FLOAT;
    case llvm::Type::DoubleTyID:
      return C_DOUBLE;
    case llvm::Type::StructTyID:
      return getOrRegisterStruct(dyn_cast<StructType>(type), dl);
    default:
      break;
  }
  return UNKNOWN;
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

  size_t n = type->getStructNumElements();

  std::vector<size_t> offsets;
  std::vector<size_t> arraySizes;
  std::vector<TypeInfo> memberTypeInfo;

  const StructLayout* layout = dl.getStructLayout(type);

  for (uint32_t i = 0; i < n; i++) {
    auto memberType = type->getStructElementType(i);
    int memberID = UNKNOWN;
    TypeKind kind;

    size_t arraySize = 1;

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
      LOG_ERROR("Encountered unhandled type: " << typeart::util::dump(*memberType));
      assert(false && "Encountered unhandled type");
    }

    size_t offset = layout->getElementOffset(i);
    offsets.push_back(offset);
    arraySizes.push_back(arraySize);
    memberTypeInfo.push_back({kind, memberID});
  }

  size_t numBytes = layout->getSizeInBytes();

  StructTypeInfo structInfo{id, name, numBytes, n, offsets, memberTypeInfo, arraySizes};
  typeDB.registerStruct(structInfo);

  structMap.insert({name, id});
  return id;
}
}  // namespace typeart

//
// Created by sebastian on 27.04.18.
//

#include "TypeManager.h"
#include "TypeIO.h"
#include "support/Logger.h"
#include "support/TypeUtil.h"
#include "support/Util.h"

#include <iostream>

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
        return TA_INT8;
      } else if (type == Type::getInt16Ty(c)) {
        return TA_INT16;
      } else if (type == Type::getInt32Ty(c)) {
        return TA_INT32;
      } else if (type == Type::getInt64Ty(c)) {
        return TA_INT64;
      } else {
        return TA_UNKNOWN_TYPE;
      }
    // TODO: Unsigned types are not supported as of now
    case llvm::Type::FloatTyID:
      return TA_FLOAT;
    case llvm::Type::DoubleTyID:
      return TA_DOUBLE;
    case llvm::Type::FP128TyID:
      return TA_FP128;
    case llvm::Type::X86_FP80TyID:
      return TA_X86_FP80;
    case llvm::Type::PPC_FP128TyID:
      return TA_PPC_FP128;
    case llvm::Type::PointerTyID:
      return TA_PTR;
    case llvm::Type::StructTyID:
      return getOrRegisterStruct(dyn_cast<StructType>(type), dl);
    default:
      break;
  }
  return TA_UNKNOWN_TYPE;
}

int TypeManager::getOrRegisterStruct(llvm::StructType* type, const llvm::DataLayout& dl) {
  namespace tu = typeart::util;
  const auto getName = [](auto type) -> std::string {
    if (type->isLiteral()) {
      return "LiteralS" + std::to_string(reinterpret_cast<long int>(type));
    }
    return type->getStructName();
  };

  auto name = getName(type);
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
  int id = TA_NUM_RESERVED_IDS + structCount;
  structCount++;

  size_t n = type->getStructNumElements();

  std::vector<size_t> offsets;
  std::vector<size_t> arraySizes;
  std::vector<int> memberTypeIDs;

  const StructLayout* layout = dl.getStructLayout(type);

  for (uint32_t i = 0; i < n; i++) {
    llvm::Type* memberType = type->getStructElementType(i);
    int memberID = TA_UNKNOWN_TYPE;

    size_t arraySize = 1;

    if (memberType->isArrayTy()) {
      arraySize = tu::type::getArrayLengthFlattened(memberType);
      memberType = tu::type::getArrayElementType(memberType);
    }

    if (memberType->isStructTy()) {
      if (getName(llvm::dyn_cast<StructType>(memberType)) == name) {
        memberID = id;
      } else {
        // TODO: Infinite cycle possible?
        memberID = getOrRegisterType(memberType, dl);
      }
    } /*else if (memberType->isPointerTy()) {
      memberID = TA_PTR;
    }*/
    else if (memberType->isSingleValueType() || memberType->isPointerTy()) {
      memberID = getOrRegisterType(memberType, dl);
    } else {
      // TODO: Any other types?
      // clang-format off
      LOG_ERROR("In struct: " << tu::dump(*type)
                  << ": Encountered unhandled type: " << tu::dump(*memberType)
                  << " with type id: " << memberType->getTypeID());
      // clang-format on
      assert(false && "Encountered unhandled type");
    }

    size_t offset = layout->getElementOffset(i);
    offsets.push_back(offset);
    arraySizes.push_back(arraySize);
    memberTypeIDs.push_back(memberID);
  }

  size_t numBytes = layout->getSizeInBytes();

  StructTypeInfo structInfo{id, name, numBytes, n, offsets, memberTypeIDs, arraySizes};
  typeDB.registerStruct(structInfo);

  structMap.insert({name, id});
  return id;
}
}  // namespace typeart

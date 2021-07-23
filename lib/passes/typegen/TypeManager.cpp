//
// Created by sebastian on 27.04.18.
//

#include "TypeManager.h"

#include "support/Logger.h"
#include "support/TypeUtil.h"
#include "support/Util.h"
#include "typelib/TypeIO.h"
#include "typelib/TypeInterface.h"

#include "llvm/ADT/None.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

namespace typeart {

using namespace llvm;

struct StructHandler {
  const llvm::StringMap<int>* m_struct_map;
  const TypeDB* m_type_db;

  [[nodiscard]] static std::string getName(const StructType* type) {
    if (type->isLiteral()) {
      return "LiteralS" + std::to_string(reinterpret_cast<long int>(type));
    }
    return type->getStructName();
  }

  [[nodiscard]] llvm::Optional<int> getIDFor(const llvm::StructType* type) const {
    const auto name = StructHandler::getName(type);
    if (auto it = m_struct_map->find(name); it != m_struct_map->end()) {
      const auto type_id = it->second;
      if (!m_type_db->isUserDefinedType(type_id)) {
        LOG_ERROR("Expected user defined struct type " << name << " for type id: " << type_id);
        return TA_UNKNOWN_TYPE;
      }
      return type_id;
    }
    return None;
  }
};

struct VectorHandler {
  // To avoid problems with padding bytes due to alignment, vector types are represented as structs rather than static
  // arrays. They are given special names and are marked with a TA_VEC flag to avoid confusion.

  const llvm::StringMap<int>* m_struct_map;
  const TypeDB* m_type_db;

  [[nodisard]] llvm::Optional<int> getElementID(llvm::VectorType* type, const DataLayout& dl, const TypeManager& m) {
    auto elementType = type->getVectorElementType();

    // Should never happen, as vectors are first class types.
    assert(!elementType->isAggregateType() && "Unexpected vector type encountered: vector of aggregate type.");

    int elementId = m.getTypeID(elementType, dl);
    if (elementId == TA_UNKNOWN_TYPE) {
      LOG_ERROR("Encountered vector of unknown type" << util::dump(*type));
      return TA_UNKNOWN_TYPE;
    }

    return None;
  }

  [[nodiscard]] llvm::Optional<std::string> getName(llvm::VectorType* type, const DataLayout& dl,
                                                    const TypeManager& m) {
    const auto elementId = getElementID(type, dl, m);
    if (!elementId || elementId.getValue() == TA_UNKNOWN_TYPE) {
      return None;
    }

    size_t vectorSize = type->getVectorNumElements();
    auto elementName  = m_type_db->getTypeName(elementId.getValue());
    auto name         = "vec" + std::to_string(vectorSize) + ":" + elementName;

    return name;
  }

  [[nodiscard]] llvm::Optional<int> getIDFor(llvm::VectorType* type, const DataLayout& dl, const TypeManager& m) {
    const auto name = getName(type, dl, m);
    if (!name) {
      return TA_UNKNOWN_TYPE;
    }

    if (auto it = m_struct_map->find(name.getValue()); it != m_struct_map->end()) {
      if (!m_type_db->isVectorType(it->second)) {
        LOG_ERROR("Expected vector type: " << name.getValue());
        return TA_UNKNOWN_TYPE;
      }
      return it->second;
    }

    return None;
  }
};

TypeManager::TypeManager(std::string file) : file(std::move(file)), structCount(0) {
}

std::pair<bool, std::error_code> TypeManager::load() {
  TypeIO cio(&typeDB);
  std::error_code error;
  if (cio.load(file, error)) {
    structMap.clear();
    for (const auto& structInfo : typeDB.getStructList()) {
      structMap.insert({structInfo.name, structInfo.id});
    }
    structCount = structMap.size();
    return {true, error};
  }
  return {false, error};
}

std::pair<bool, std::error_code> TypeManager::store() {
  std::error_code error;
  TypeIO cio(&typeDB);
  const bool ret = cio.store(file, error);
  return {ret, error};
}

llvm::Optional<typeart_builtin_type> get_builtin_typeid(llvm::Type* type) {
  auto& c = type->getContext();
  switch (type->getTypeID()) {
    case llvm::Type::IntegerTyID: {
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
    }
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
  }
  return None;
}

int TypeManager::getTypeID(llvm::Type* type, const DataLayout& dl) const {
  auto builtin_id = get_builtin_typeid(type);
  if (builtin_id) {
    return builtin_id.getValue();
  }

  switch (type->getTypeID()) {
    case llvm::Type::VectorTyID: {
      VectorHandler handle{&structMap, &typeDB};
      const auto type_id = handle.getIDFor(dyn_cast<VectorType>(type), dl, this);
      if (type_id) {
        return type_id.getValue();
      }
    }
    case llvm::Type::StructTyID: {
      StructHandler handle{&structMap, &typeDB};
      const auto type_id = handle.getIDFor(dyn_cast<StructType>(type));
      if (type_id) {
        return type_id.getValue();
      }
    }
    default:
      break;
  }

  return TA_UNKNOWN_TYPE;
}

int TypeManager::getOrRegisterType(llvm::Type* type, const llvm::DataLayout& dl) {
  auto builtin_id = get_builtin_typeid(type);
  if (builtin_id) {
    return builtin_id.getValue();
  }

  switch (type->getTypeID()) {
    case llvm::Type::VectorTyID:
      return getOrRegisterVector(dyn_cast<VectorType>(type), dl);
    case llvm::Type::StructTyID:
      return getOrRegisterStruct(dyn_cast<StructType>(type), dl);
    default:
      break;
  }
  return TA_UNKNOWN_TYPE;
}

int TypeManager::getOrRegisterVector(llvm::VectorType* type, const llvm::DataLayout& dl) {
  namespace tu = typeart::util;

  //  size_t vectorBytes = dl.getTypeAllocSize(type);
  //  size_t vectorSize  = type->getVectorNumElements();
  //
  //  auto elementType = type->getVectorElementType();
  //
  //  // Should never happen, as vectors are first class types.
  //  assert(!elementType->isAggregateType() && "Unexpected vector type encountered: vector of aggregate type.");
  //
  //  int elementId = getOrRegisterType(elementType, dl);
  //  if (elementId == TA_UNKNOWN_TYPE) {
  //    LOG_ERROR("Encountered vector of unknown type" << util::dump(*type));
  //    return elementId;
  //  }
  //  auto elementName = typeDB.getTypeName(elementId);
  //  auto name        = "vec" + std::to_string(vectorSize) + ":" + elementName;
  //
  //  // To avoid problems with padding bytes due to alignment, vector types are represented as structs rather than
  //  static
  //  // arrays. They are given special names and are marked with a TA_VEC flag to avoid confusion.
  //
  //  // Look up name
  //  if (auto it = structMap.find(name); it != structMap.end()) {
  //    if (!typeDB.isVectorType(it->second)) {
  //      LOG_ERROR("Expected vector type: " << name);
  //      return TA_UNKNOWN_TYPE;
  //    }
  //    return it->second;
  //  }

  VectorHandler handler{&structMap, &typeDB};
  const auto type_id = handler.getIDFor(type, dl, *this);
  if (type_id) {
    return type_id.getValue();
  }

  // Type is not registered - reserve new ID and create struct info object

  int id = reserveNextId();

  std::vector<int> memberTypeIDs{elementId};
  std::vector<size_t> arraySizes{vectorSize};
  std::vector<size_t> offsets{0};

  size_t elementSize = tu::type::getTypeSizeInBytes(elementType, dl);
  size_t usableBytes = vectorSize * elementSize;

  // Add padding bytes explicitly
  if (vectorBytes > usableBytes) {
    size_t padding = vectorBytes - usableBytes;
    memberTypeIDs.push_back(TA_INT8);
    arraySizes.push_back(padding);
    offsets.push_back(usableBytes);
  }

  StructTypeInfo vecTypeInfo{id,      name,          vectorBytes, memberTypeIDs.size(),
                             offsets, memberTypeIDs, arraySizes,  StructTypeFlag::LLVM_VECTOR};
  typeDB.registerStruct(vecTypeInfo);
  structMap.insert({name, id});
  return id;
}

int TypeManager::getOrRegisterStruct(llvm::StructType* type, const llvm::DataLayout& dl) {
  namespace tu = typeart::util;

  StructHandler handle{&structMap, &typeDB};
  const auto type_id = handle.getIDFor(type);
  if (type_id) {
    return type_id.getValue();
  }

  const auto name = StructHandler::getName(type);
  // Get next ID and register struct
  int id = reserveNextId();

  size_t n = type->getStructNumElements();

  std::vector<size_t> offsets;
  std::vector<size_t> arraySizes;
  std::vector<int> memberTypeIDs;

  const StructLayout* layout = dl.getStructLayout(type);

  for (unsigned i = 0; i < n; ++i) {
    llvm::Type* memberType = type->getStructElementType(i);
    int memberID           = TA_UNKNOWN_TYPE;
    size_t arraySize       = 1;

    if (memberType->isArrayTy()) {
      // Note that clang does not allow VLAs inside of structs (GCC does)
      arraySize  = tu::type::getArrayLengthFlattened(memberType);
      memberType = tu::type::getArrayElementType(memberType);
    }

    if (memberType->isStructTy()) {
      if (StructHandler::getName(llvm::dyn_cast<StructType>(memberType)) == name) {
        memberID = id;
      } else {
        memberID = getOrRegisterType(memberType, dl);
      }
    } else if (memberType->isSingleValueType() || memberType->isPointerTy()) {
      memberID = getOrRegisterType(memberType, dl);
    } else {
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

  StructTypeInfo structInfo{id, name, numBytes, n, offsets, memberTypeIDs, arraySizes, StructTypeFlag::USER_DEFINED};
  typeDB.registerStruct(structInfo);

  structMap.insert({name, id});
  return id;
}

int TypeManager::reserveNextId() {
  int id = TA_NUM_RESERVED_IDS + structCount;
  structCount++;
  return id;
}

}  // namespace typeart

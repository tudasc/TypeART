#include "TypeManager.h"

#include "StructTypeHandler.h"
#include "VectorTypeHandler.h"
#include "support/Logger.h"
#include "support/TypeUtil.h"
#include "support/Util.h"
#include "typelib/TypeIO.h"
#include "typelib/TypeInterface.h"

#include "llvm/ADT/None.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <utility>
#include <vector>

namespace typeart {

std::unique_ptr<TypeGenerator> make_typegen(const std::string& file) {
  return std::make_unique<TypeManager>(file);
}

using namespace llvm;

llvm::Optional<typeart_builtin_type> get_builtin_typeid(llvm::Type* type) {
  auto& c = type->getContext();

  switch (type->getTypeID()) {
    case llvm::Type::IntegerTyID: {
      if (type == Type::getInt8Ty(c)) {
        return TA_INT8;
      }
      if (type == Type::getInt16Ty(c)) {
        return TA_INT16;
      }
      if (type == Type::getInt32Ty(c)) {
        return TA_INT32;
      }
      if (type == Type::getInt64Ty(c)) {
        return TA_INT64;
      }
      return TA_UNKNOWN_TYPE;
    }
    case llvm::Type::HalfTyID:
      return TA_HALF;
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
    default:
      return None;
  }
}

int TypeManager::getOrRegisterVector(llvm::VectorType* type, const llvm::DataLayout& dl) {
  namespace tu = typeart::util;

  VectorTypeHandler handler{&structMap, &typeDB, type, dl, *this};
  const auto type_id = handler.getID();
  if (type_id) {
    return type_id.getValue();
  }

  // Type is not registered - reserve new ID and create struct info object:

  auto element_data = handler.getElementData();
  if (!element_data) {
    return TA_UNKNOWN_TYPE;
  }

  auto vector_data = handler.getVectorData();
  if (!vector_data) {
    return TA_UNKNOWN_TYPE;
  }

  const int id = reserveNextId();

  const auto [element_id, element_type, element_name] = element_data.getValue();
  const auto [vector_name, vector_bytes, vector_size] = vector_data.getValue();

  std::vector<int> memberTypeIDs{element_id};
  std::vector<size_t> arraySizes{vector_size};
  std::vector<size_t> offsets{0};

  size_t elementSize = tu::type::getTypeSizeInBytes(element_type, dl);
  size_t usableBytes = vector_size * elementSize;

  // Add padding bytes explicitly
  if (vector_bytes > usableBytes) {
    size_t padding = vector_bytes - usableBytes;
    memberTypeIDs.push_back(TA_INT8);
    arraySizes.push_back(padding);
    offsets.push_back(usableBytes);
  }

  StructTypeInfo vecTypeInfo{id,      vector_name,   vector_bytes, memberTypeIDs.size(),
                             offsets, memberTypeIDs, arraySizes,   StructTypeFlag::LLVM_VECTOR};
  typeDB.registerStruct(vecTypeInfo);
  structMap.insert({vector_name, id});
  return id;
}

TypeManager::TypeManager(std::string file) : file(std::move(file)), structCount(0) {
}

std::pair<bool, std::error_code> TypeManager::load() {
  //  TypeIO cio(&typeDB);
  // std::error_code error;
  auto loaded        = io::load(&typeDB, file);
  std::error_code ec = loaded.getError();
  if (ec) {
    return {false, ec};
  }
  structMap.clear();
  for (const auto& structInfo : typeDB.getStructList()) {
    structMap.insert({structInfo.name, structInfo.id});
  }
  structCount = structMap.size();
  return {true, ec};
}

std::pair<bool, std::error_code> TypeManager::store() const {
  auto stored        = io::store(&typeDB, file);
  std::error_code ec = stored.getError();
  return {!static_cast<bool>(ec), ec};
}

int TypeManager::getTypeID(llvm::Type* type, const DataLayout& dl) const {
  auto builtin_id = get_builtin_typeid(type);
  if (builtin_id) {
    return builtin_id.getValue();
  }

  switch (type->getTypeID()) {
    case llvm::Type::VectorTyID: {
      VectorTypeHandler handle{&structMap, &typeDB, dyn_cast<VectorType>(type), dl, *this};
      const auto type_id = handle.getID();
      if (type_id) {
        return type_id.getValue();
      }
      break;
    }
    case llvm::Type::StructTyID: {
      StructTypeHandler handle{&structMap, &typeDB, dyn_cast<StructType>(type)};
      const auto type_id = handle.getID();
      if (type_id) {
        return type_id.getValue();
      }
      break;
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

int TypeManager::getOrRegisterStruct(llvm::StructType* type, const llvm::DataLayout& dl) {
  namespace tu = typeart::util;

  StructTypeHandler handle{&structMap, &typeDB, type};
  const auto type_id = handle.getID();
  if (type_id) {
    return type_id.getValue();
  }

  const auto name = handle.getName();

  // Get next ID and register struct:
  const int id = reserveNextId();

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
      if (StructTypeHandler::getName(llvm::dyn_cast<StructType>(memberType)) == name) {
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
  int id = static_cast<int>(TA_NUM_RESERVED_IDS) + structCount;
  structCount++;
  return id;
}

}  // namespace typeart

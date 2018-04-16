#include "TypeMapping.h"

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Type.h"

using namespace llvm;

namespace must {

SimpleTypeMapping::SimpleTypeMapping() {
}

int SimpleTypeMapping::getTypeId(llvm::Type* type) {
  int size = type->getScalarSizeInBits();
  if (type->isIntegerTy()) {
    return createId(INT_BITS, size);
  }
  if (type->isFloatingPointTy()) {
    return createId(FLOAT_BITS, size);
  }
  if (type->isStructTy()) {
    return getIdForStruct(type);
  }
  if (type->isPointerTy()) {
    return createId(OTHER_BITS, PTR_ID);
  }
  return createId(OTHER_BITS, OTHER_ID);  // TODO: Okay to lump all pointers together?
}

int SimpleTypeMapping::getIdForStruct(llvm::Type* structTy) {
  auto name = structTy->getStructName();
  auto it = std::find(structs.begin(), structs.end(), name);
  int id;
  if (it == structs.end()) {
    id = (int)structs.size();
    structs.push_back(name);  // TODO: Name alone sufficient?
  } else {
    id = it - structs.begin();
  }
  return createId(STRUCT_BITS, id);
}

int SimpleTypeMapping::createId(int baseTypeBits, int uniqueTypeBits) {
  return (baseTypeBits & 0x3) | (uniqueTypeBits << 2);
}

}  // namespace must

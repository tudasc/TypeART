#ifndef _LIB_TYPEMAPPING_H
#define _LIB_TYPEMAPPING_H

#include <vector>

namespace llvm {
class Type;
class StringRef;
}

namespace must {

class TypeMapping {
 public:
  virtual int getTypeId(llvm::Type* type) = 0;
};

class SimpleTypeMapping : public TypeMapping {
 public:
  SimpleTypeMapping();

  int getTypeId(llvm::Type* type) override;

 private:
  int getIdForStruct(llvm::Type* structTy);
  int createId(int baseTypeBits, int uniqueTypeBits);

  std::vector<llvm::StringRef> structs;

  static const int INT_BITS = 0;
  static const int FLOAT_BITS = 1;
  static const int STRUCT_BITS = 2;
  static const int OTHER_BITS = 3;
};

}  // namespace must

#endif
#ifndef LLVM_MUST_SUPPORT_TYPEMANAGER_H
#define LLVM_MUST_SUPPORT_TYPEMANAGER_H

#include <llvm/IR/Type.h>

#include <TypeDB.h>
#include <llvm/IR/DataLayout.h>

#include <map>

namespace typeart {

class TypeManager {
 public:
  explicit TypeManager(std::string file);

  bool load();

  bool store();

  int getOrRegisterType(llvm::Type* type, const llvm::DataLayout& dl);

 private:
  int getOrRegisterStruct(llvm::StructType* type, const llvm::DataLayout& dl);
  int getOrRegisterVector(llvm::VectorType* type, const llvm::DataLayout& dl);

  int reserveNextId();

  // TypeInfo getTypeInfo(llvm::Type* type);

  std::string file;

  TypeDB typeDB;
  std::map<std::string, int> structMap;

  size_t structCount;
};
}  // namespace typeart

#endif  // LLVM_MUST_SUPPORT_TYPEMANAGER_H

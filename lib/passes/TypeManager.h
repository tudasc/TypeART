#ifndef LLVM_MUST_SUPPORT_TYPEMANAGER_H
#define LLVM_MUST_SUPPORT_TYPEMANAGER_H

#include "typelib/TypeDB.h"

#include <llvm/ADT/StringMap.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Type.h>

namespace typeart {

class TypeManager {
  std::string file;
  TypeDB typeDB;
  llvm::StringMap<int> structMap;
  size_t structCount;

 public:
  explicit TypeManager(std::string file);
  bool load();
  bool store();
  int getOrRegisterType(llvm::Type* type, const llvm::DataLayout& dl);

 private:
  int getOrRegisterStruct(llvm::StructType* type, const llvm::DataLayout& dl);
  int getOrRegisterVector(llvm::VectorType* type, const llvm::DataLayout& dl);
  int reserveNextId();
};
}  // namespace typeart

#endif  // LLVM_MUST_SUPPORT_TYPEMANAGER_H

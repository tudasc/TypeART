#ifndef LLVM_MUST_SUPPORT_TYPEMANAGER_H
#define LLVM_MUST_SUPPORT_TYPEMANAGER_H

#include <llvm/IR/Type.h>

#include <TypeConfig.h>
#include <llvm/IR/DataLayout.h>

namespace must {

class TypeManager {
 public:
  TypeManager();

  bool load(std::string file);

  bool store(std::string file);

  int getOrRegisterType(llvm::Type* type, const llvm::DataLayout& dl);

 private:
  int getOrRegisterStruct(llvm::StructType* type, const llvm::DataLayout& dl);

  // TypeInfo getTypeInfo(llvm::Type* type);

  TypeConfig typeConfig;

  std::map<std::string, int> structMap;
  int structCount;
};
}

#endif  // LLVM_MUST_SUPPORT_TYPEMANAGER_H

#ifndef LLVM_MUST_SUPPORT_TYPEMANAGER_H
#define LLVM_MUST_SUPPORT_TYPEMANAGER_H

#include "TypeGenerator.h"
#include "typelib/TypeDB.h"

#include "llvm/ADT/StringMap.h"

#include <cstddef>
#include <string>

namespace llvm {
class DataLayout;
class StructType;
class Type;
class VectorType;
}  // namespace llvm

namespace typeart {

class TypeManager final : public TypeGenerator {
  std::string file;
  TypeDB typeDB;
  llvm::StringMap<int> structMap;
  size_t structCount;

 public:
  explicit TypeManager(std::string file);
  [[nodiscard]] std::pair<bool, std::error_code> load() override;
  [[nodiscard]] std::pair<bool, std::error_code> store() override;
  [[nodiscard]] int getOrRegisterType(llvm::Type* type, const llvm::DataLayout& dl) override;
  [[nodiscard]] int getTypeID(llvm::Type* type, const llvm::DataLayout& dl) const override;

 private:
  [[nodiscard]] int getOrRegisterStruct(llvm::StructType* type, const llvm::DataLayout& dl);
  [[nodiscard]] int getOrRegisterVector(llvm::VectorType* type, const llvm::DataLayout& dl);
  [[nodiscard]] int reserveNextId();
};

}  // namespace typeart

#endif  // LLVM_MUST_SUPPORT_TYPEMANAGER_H

#ifndef TYPEART_TYPEGENERATOR_H
#define TYPEART_TYPEGENERATOR_H

#include <memory>
#include <string>
#include <system_error>
#include <utility>

namespace llvm {
class Type;
class DataLayout;
}  // namespace llvm

namespace typeart {

class TypeGenerator {
 public:
  [[nodiscard]] virtual int getOrRegisterType(llvm::Type* type, const llvm::DataLayout& layout) = 0;

  [[nodiscard]] virtual int getTypeID(llvm::Type* type, const llvm::DataLayout& layout) const = 0;

  [[nodiscard]] virtual std::pair<bool, std::error_code> load() = 0;

  [[nodiscard]] virtual std::pair<bool, std::error_code> store() const = 0;

  virtual ~TypeGenerator() = default;
};

// This doesn't immediately load the file, call load/store after
std::unique_ptr<TypeGenerator> make_typegen(const std::string& file);

}  // namespace typeart

#endif  // TYPEART_TYPEGENERATOR_H

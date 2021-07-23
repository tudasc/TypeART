#ifndef TYPEART_TYPEDATABASE_H
#define TYPEART_TYPEDATABASE_H

#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace typeart {

enum class StructTypeFlag : int { USER_DEFINED = 1, LLVM_VECTOR = 2 };

struct StructTypeInfo final {
  int id;
  std::string name;
  size_t extent;
  size_t num_members;
  std::vector<size_t> offsets;
  std::vector<int> member_types;
  std::vector<size_t> array_sizes;
  StructTypeFlag flag;
};

class TypeDatabase {
 public:
  virtual void registerStruct(const StructTypeInfo& structInfo) = 0;

  [[nodiscard]] virtual bool isValid(int id) const = 0;

  [[nodiscard]] virtual bool isReservedType(int id) const = 0;

  [[nodiscard]] virtual bool isBuiltinType(int id) const = 0;

  [[nodiscard]] virtual bool isStructType(int id) const = 0;

  [[nodiscard]] virtual bool isUserDefinedType(int id) const = 0;

  [[nodiscard]] virtual bool isVectorType(int id) const = 0;

  [[nodiscard]] virtual const std::string& getTypeName(int id) const = 0;

  [[nodiscard]] virtual const StructTypeInfo* getStructInfo(int id) const = 0;

  [[nodiscard]] virtual size_t getTypeSize(int id) const = 0;

  [[nodiscard]] virtual const std::vector<StructTypeInfo>& getStructList() const = 0;

  virtual ~TypeDatabase() = default;
};

std::pair<std::unique_ptr<TypeDatabase>, std::error_code> make_database(const std::string& file);

}  // namespace typeart

#endif  // TYPEART_TYPEDATABASE_H

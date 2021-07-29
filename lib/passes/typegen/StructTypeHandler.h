#ifndef TYPEART_STRUCTTYPEHANDLER_H
#define TYPEART_STRUCTTYPEHANDLER_H

#include "llvm/ADT/None.h"
#include "llvm/ADT/StringMap.h"

#include <string>

namespace llvm {
class StructType;
}  // namespace llvm

namespace typeart {

class TypeDatabase;

struct StructTypeHandler {
  const llvm::StringMap<int>* m_struct_map;
  const TypeDatabase* m_type_db;
  llvm::StructType* type;

  [[nodiscard]] static std::string getName(llvm::StructType* type);

  [[nodiscard]] std::string getName() const;

  [[nodiscard]] llvm::Optional<int> getID() const;
};

}  // namespace typeart

#endif  // TYPEART_STRUCTTYPEHANDLER_H

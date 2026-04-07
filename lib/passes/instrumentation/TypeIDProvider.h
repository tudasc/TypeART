#ifndef TYPEART_TYPEIDPROVIDER_H
#define TYPEART_TYPEIDPROVIDER_H

// #include "TypeARTFunctions.h"
// #include "instrumentation/TypeARTFunctions.h"
#include "typegen/TypeGenerator.h"
#include "typelib/TypeDatabase.h"

#include <cstdint>
#include <memory>

namespace llvm {
class Value;
class Module;
}  // namespace llvm

namespace typeart {

enum class TypeSerializationImplementation : uint8_t { FILE, INLINE, HYBRID };

namespace config {
class Configuration;
}
class TAFunctionQuery;

class TypeRegistry {
 public:
  [[nodiscard]] virtual llvm::Value* getOrRegister(llvm::Value* type_id_const) = 0;
  virtual void registerModule(const ModuleData&);
  virtual ~TypeRegistry() = default;
};

std::unique_ptr<TypeRegistry> get_type_id_handler(llvm::Module& m, const TypeDatabase* type_db,
                                                  const config::Configuration& configuration,
                                                  const TAFunctionQuery* f_query);

}  // namespace typeart

#endif  // TYPEART_TYPEIDPROVIDER_H

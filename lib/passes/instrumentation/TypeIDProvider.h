#ifndef LIB_PASSES_INSTRUMENTATION_MODULETYPEREGISTRY
#define LIB_PASSES_INSTRUMENTATION_MODULETYPEREGISTRY

#include "TypeDatabase.h"
#include "TypeGenerator.h"

#include <memory>

namespace llvm {
class Value;
class Module;
}  // namespace llvm

namespace typeart {

namespace config {
class Configuration;
}

class TypeRegistry {
 public:
  virtual llvm::Value* getOrRegister(llvm::Value* type_id_const) = 0;
  virtual void registerModule(const ModuleData&);
  virtual ~TypeRegistry() = default;
};

std::unique_ptr<TypeRegistry> get_type_id_handler(llvm::Module& m, const TypeDatabase* type_db,
                                                  const config::Configuration& configuration);

}  // namespace typeart

#endif /* LIB_PASSES_INSTRUMENTATION_MODULETYPEREGISTRY */

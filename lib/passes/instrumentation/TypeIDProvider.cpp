#include "TypeIDProvider.h"

#include "configuration/Configuration.h"

#include <memory>

namespace typeart {

class TypeRegistryNoOp final : public TypeRegistry {
 public:
  llvm::Value* getOrRegister(llvm::Value* type_id_const) override {
    return type_id_const;
  }
};

std::unique_ptr<TypeRegistry> get_type_id_handler(llvm::Module& m, const config::Configuration& configuration) {
  return std::make_unique<TypeRegistryNoOp>();
}

}  // namespace typeart
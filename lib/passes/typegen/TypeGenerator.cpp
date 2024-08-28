#include "TypeGenerator.h"

#include "TypeIDGenerator.h"
#include "dimeta/DimetaTypeGen.h"
#include "ir/IRTypeGen.h"
#include "support/Logger.h"
#include "typelib/TypeDB.h"
#include "typelib/TypeIO.h"
#include "typelib/TypeInterface.h"

namespace typeart::types {

TypeIDGenerator::TypeIDGenerator(std::string file_) : file(std::move(file_)) {
}

std::pair<bool, std::error_code> TypeIDGenerator::load() {
  auto loaded        = typeart::io::load(&typeDB, file);
  std::error_code ec = loaded.getError();
  if (ec) {
    return {false, ec};
  }
  structMap.clear();
  for (const auto& structInfo : typeDB.getStructList()) {
    structMap.insert({structInfo.name, structInfo.type_id});
  }
  structCount = structMap.size();
  return {true, ec};
}

std::pair<bool, std::error_code> TypeIDGenerator::store() const {
  auto stored        = typeart::io::store(&typeDB, file);
  std::error_code ec = stored.getError();
  return {!static_cast<bool>(ec), ec};
}

int TypeIDGenerator::reserveNextTypeId() {
  int id = static_cast<int>(TYPEART_NUM_RESERVED_IDS) + structCount;
  structCount++;
  return id;
}

const TypeDatabase& TypeIDGenerator::getTypeDatabase() const {
  return this->typeDB;
}

}  // namespace typeart::types

namespace typeart {
std::unique_ptr<TypeGenerator> make_typegen(std::string_view file, TypegenImplementation impl) {
  switch (impl) {
    case typeart::TypegenImplementation::DIMETA:
      LOG_DEBUG("Loading Dimeta type parser")
      return types::make_dimeta_typeidgen(file);
    default:
      break;
  }
  LOG_DEBUG("Loading IR type parser")
  return make_ir_typeidgen(file);
}
}  // namespace typeart

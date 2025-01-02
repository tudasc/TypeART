// TypeART library
//
// Copyright (c) 2017-2025 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#include "TypeGenerator.h"

#include "TypeIDGenerator.h"
#include "dimeta/DimetaTypeGen.h"
#include "ir/IRTypeGen.h"
#include "support/Logger.h"
#include "typelib/TypeDB.h"
#include "typelib/TypeIO.h"
#include "typelib/TypeInterface.h"

#include <memory>

namespace typeart::types {

TypeIDGenerator::TypeIDGenerator(std::string file_, std::unique_ptr<TypeDatabase> database_)
    : file(std::move(file_)), typeDB(std::move(database_)) {
}

std::pair<bool, std::error_code> TypeIDGenerator::load() {
  auto loaded        = typeart::io::load(typeDB.get(), file);
  std::error_code ec = loaded.getError();
  if (ec) {
    return {false, ec};
  }
  structMap.clear();
  for (const auto& structInfo : typeDB->getStructList()) {
    structMap.insert({structInfo.name, structInfo.type_id});
  }
  structCount = structMap.size();
  return {true, ec};
}

std::pair<bool, std::error_code> TypeIDGenerator::store() const {
  auto stored        = typeart::io::store(typeDB.get(), file);
  std::error_code ec = stored.getError();
  return {!static_cast<bool>(ec), ec};
}

int TypeIDGenerator::reserveNextTypeId() {
  int id = static_cast<int>(TYPEART_NUM_RESERVED_IDS) + structCount;
  structCount++;
  return id;
}

const TypeDatabase& TypeIDGenerator::getTypeDatabase() const {
  return *this->typeDB.get();
}

void TypeIDGenerator::registerModule(const ModuleData&) {
}

}  // namespace typeart::types

namespace typeart {
std::unique_ptr<TypeGenerator> make_typegen(std::string_view file, TypegenImplementation impl) {
  auto database = std::make_unique<TypeDB>();
  switch (impl) {
    case typeart::TypegenImplementation::DIMETA:
      LOG_DEBUG("Loading Dimeta type parser")
      return types::make_dimeta_typeidgen(file, std::move(database));
    default:
      break;
  }
  LOG_DEBUG("Loading IR type parser")
  return make_ir_typeidgen(file, std::move(database));
}
}  // namespace typeart

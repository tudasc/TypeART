// TypeART library
//
// Copyright (c) 2017-2022 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#include "StructTypeHandler.h"

#include "support/Logger.h"
#include "support/Util.h"
#include "typelib/TypeDatabase.h"
#include "typelib/TypeInterface.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Support/raw_ostream.h"

namespace typeart {

std::string StructTypeHandler::getName(llvm::StructType* type) {
  if (type->isLiteral()) {
    return "LiteralS" + std::to_string(reinterpret_cast<long int>(type));
  }
  return std::string{type->getStructName()};
}

std::string StructTypeHandler::getName() const {
  return getName(type);
}

llvm::Optional<int> StructTypeHandler::getID() const {
  const auto name = StructTypeHandler::getName(type);
  if (auto it = m_struct_map->find(name); it != m_struct_map->end()) {
    const auto type_id = it->second;
    if (!m_type_db->isUserDefinedType(type_id)) {
      LOG_ERROR("Expected user defined struct type " << name << " for type id: " << type_id);
      return TYPEART_UNKNOWN_TYPE;
    }
    return type_id;
  }
  return llvm::None;
}

}  // namespace typeart
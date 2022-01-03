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

#include "VectorTypeHandler.h"

#include "TypeGenerator.h"
#include "support/Logger.h"
#include "support/Util.h"
#include "typelib/TypeDatabase.h"
#include "typelib/TypeInterface.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <vector>

namespace typeart {

using namespace llvm;

namespace compat {
auto num_elements(llvm::VectorType* type) {
#if LLVM_VERSION_MAJOR < 11
  return type->getVectorNumElements();
#else
  auto* fixed_vec = dyn_cast<llvm::FixedVectorType>(type);
  assert(fixed_vec);
  return fixed_vec->getNumElements();
#endif
}
auto type_elements(llvm::VectorType* type) {
#if LLVM_VERSION_MAJOR < 11
  return type->getVectorElementType();
#else
  auto* fixed_vec = dyn_cast<llvm::FixedVectorType>(type);
  assert(fixed_vec);
  return fixed_vec->getElementType();
#endif
}
}  // namespace compat

Type* VectorTypeHandler::getElementType(llvm::VectorType* type) {
  auto element_type = compat::type_elements(type);

  // Should never happen, as vectors are first class types.
  assert(!element_type->isAggregateType() && "Unexpected vector type encountered: vector of aggregate type.");

  return element_type;
}

llvm::Optional<VectorTypeHandler::VectorData> VectorTypeHandler::getVectorData() const {
  size_t vectorBytes = dl.getTypeAllocSize(type);

  size_t vectorSize = compat::num_elements(type);
  auto element_data = getElementData();

  if (!element_data) {
    LOG_DEBUG("Element data empty for: " << util::dump(*type))
    return None;
  }

  auto vec_name = getName(element_data.getValue());

  return VectorData{vec_name, vectorBytes, vectorSize};
}

llvm::Optional<VectorTypeHandler::ElementData> VectorTypeHandler::getElementData() const {
  const auto element_id = getElementID();
  if (!element_id || element_id.getValue() == TYPEART_UNKNOWN_TYPE) {
    LOG_WARNING("Unknown vector element id.")
    return None;
  }

  auto element_name = m_type_db->getTypeName(element_id.getValue());
  auto element_type = VectorTypeHandler::getElementType(type);
  return ElementData{element_id.getValue(), element_type, element_name};
}

llvm::Optional<int> VectorTypeHandler::getElementID() const {
  auto element_type     = getElementType(type);
  const auto element_id = m.getTypeID(element_type, dl);

  if (element_id == TYPEART_UNKNOWN_TYPE) {
    LOG_ERROR("Encountered vector of unknown type" << util::dump(*type));
    return TYPEART_UNKNOWN_TYPE;
  }

  return element_id;
}

std::string VectorTypeHandler::getName(const ElementData& data) const {
  size_t vectorSize = compat::num_elements(type);
  auto name         = "vec" + std::to_string(vectorSize) + ":" + data.element_name;

  return name;
}

llvm::Optional<int> VectorTypeHandler::getID() const {
  auto element_data = getElementData();

  if (!element_data) {
    LOG_ERROR("Cannot determine element data for " << util::dump(*type))
    return TYPEART_UNKNOWN_TYPE;
  }

  const auto name = getName(element_data.getValue());

  if (auto it = m_struct_map->find(name); it != m_struct_map->end()) {
    if (!m_type_db->isVectorType(it->second)) {
      LOG_ERROR("Expected vector type for name:" << name << " Vector: " << util::dump(*type))
      return TYPEART_UNKNOWN_TYPE;
    }
    return it->second;
  }

  return None;
}

}  // namespace typeart

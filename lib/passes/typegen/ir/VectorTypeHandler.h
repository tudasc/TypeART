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

#ifndef TYPEART_VECTORTYPEHANDLER_H
#define TYPEART_VECTORTYPEHANDLER_H

#include "llvm/ADT/StringMap.h"

#include <optional>
#include <string>

namespace llvm {
class Type;
class VectorType;
class DataLayout;
}  // namespace llvm

namespace typeart {

class TypeDatabase;
class TypeManager;

struct VectorTypeHandler {
  // To avoid problems with padding bytes due to alignment, vector types are represented as structs rather than static
  // arrays. They are given special names and are marked with a TA_VEC flag to avoid confusion.

  const llvm::StringMap<int>* m_struct_map;
  const TypeDatabase* m_type_db;

  llvm::VectorType* type;
  const llvm::DataLayout& dl;
  const TypeManager& m;

  struct VectorData {
    std::string vec_name;
    size_t vector_bytes{0};
    size_t vector_size{0};
  };

  struct ElementData {
    int element_id{-1};
    llvm::Type* element_type{nullptr};
    std::string element_name;
  };

  static llvm::Type* getElementType(llvm::VectorType* type);

  [[nodiscard]] std::optional<VectorData> getVectorData() const;

  [[nodiscard]] std::optional<ElementData> getElementData() const;

  [[nodiscard]] std::optional<int> getElementID() const;

  [[nodiscard]] std::string getName(const ElementData& data) const;

  [[nodiscard]] std::optional<int> getID() const;
};

}  // namespace typeart

#endif  // TYPEART_VECTORTYPEHANDLER_H

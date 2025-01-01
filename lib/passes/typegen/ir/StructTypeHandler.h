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

#ifndef TYPEART_STRUCTTYPEHANDLER_H
#define TYPEART_STRUCTTYPEHANDLER_H

#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
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

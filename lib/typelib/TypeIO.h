// TypeART library
//
// Copyright (c) 2017-2026 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#ifndef TYPEART_TYPEIO_H
#define TYPEART_TYPEIO_H

#include "typelib/TypeDatabase.h"

#include "llvm/Support/ErrorOr.h"

#include <string>

namespace typeart {

class TypeDB;

namespace io {
[[nodiscard]] llvm::ErrorOr<bool> load(TypeDatabase* db, const std::string& file);
[[nodiscard]] llvm::ErrorOr<bool> store(const TypeDatabase* db, const std::string& file);
}  // namespace io

}  // namespace typeart

#endif  // TYPEART_TYPEIO_H

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

#ifndef LLVM_MUST_SUPPORT_CONFIGIO_H
#define LLVM_MUST_SUPPORT_CONFIGIO_H

#include "llvm/Support/ErrorOr.h"

#include <string>

namespace typeart {

class TypeDB;

namespace io {
[[nodiscard]] llvm::ErrorOr<bool> load(TypeDB* db, const std::string& file);
[[nodiscard]] llvm::ErrorOr<bool> store(const TypeDB* db, const std::string& file);
}  // namespace io

}  // namespace typeart

#endif  // LLVM_MUST_SUPPORT_CONFIGIO_H

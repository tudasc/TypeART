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

#ifndef TYPEART_OMPUTIL_H
#define TYPEART_OMPUTIL_H

#include "Util.h"
#include "support/Logger.h"

#include "llvm/IR/Function.h"

namespace typeart::util::omp {

inline bool isOmpContext(llvm::Function* f) {
  if (f != nullptr) {
    const auto name_ = demangle(f->getName());
    llvm::StringRef fname(name_);
    return util::starts_with_any_of(fname, ".omp", "__kmpc", "__omp");
  }
  return false;
}

}  // namespace typeart::util::omp

#endif  // TYPEART_OMPUTIL_H

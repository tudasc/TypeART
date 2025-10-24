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

#ifndef TYPEART_TYPEARTFUNCTIONS_H
#define TYPEART_TYPEARTFUNCTIONS_H

#include "InstrumentationHelper.h"
#include "configuration/Configuration.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <unordered_map>

namespace llvm {
class Function;
class Type;
class Module;
}  // namespace llvm

namespace typeart {
class InstrumentationHelper;

namespace config {
class Configuration;
}

enum class IFunc : unsigned {
  heap,
  stack,
  global,
  free,
  scope,
  heap_omp,
  stack_omp,
  free_omp,
  scope_omp,
};

IFunc ifunc_for_function(IFunc general_type, llvm::Value* value);

class TAFunctionQuery {
 public:
  virtual llvm::Function* getFunctionFor(IFunc id) const = 0;
  virtual ~TAFunctionQuery()                             = default;
};

std::unique_ptr<TAFunctionQuery> declare_instrumentation_functions(llvm::Module& m,
                                                                   const config::Configuration& configuration);

}  // namespace typeart

#endif  // TYPEART_TYPEARTFUNCTIONS_H

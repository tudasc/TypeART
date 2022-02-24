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

#ifndef TYPEART_TYPEARTFUNCTIONS_H
#define TYPEART_TYPEARTFUNCTIONS_H

#include "InstrumentationHelper.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <unordered_map>

namespace llvm {
class Function;
class Type;
class Module;
}  // namespace llvm

namespace typeart {
class InstrumentationHelper;

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

class TAFunctionQuery {
 public:
  virtual llvm::Function* getFunctionFor(IFunc id) = 0;
  virtual ~TAFunctionQuery()                       = default;
};

class TAFunctions : public TAFunctionQuery {
  // densemap has problems with IFunc
  using FMap = std::unordered_map<IFunc, llvm::Function*>;
  FMap typeart_callbacks;

 public:
  TAFunctions();

  llvm::Function* getFunctionFor(IFunc id) override;
  void putFunctionFor(IFunc id, llvm::Function* f);
};

class TAFunctionDeclarator {
  llvm::Module& m;
  //  [[maybe_unused]] InstrumentationHelper& instr;
  TAFunctions& tafunc;
  llvm::StringMap<llvm::Function*> f_map;

 public:
  TAFunctionDeclarator(llvm::Module& m, InstrumentationHelper& instr, TAFunctions& tafunc);
  llvm::Function* make_function(IFunc function, llvm::StringRef basename, llvm::ArrayRef<llvm::Type*> args,
                                bool with_omp = false, bool fixed_name = true);
  const llvm::StringMap<llvm::Function*>& getFunctionMap() const;
  virtual ~TAFunctionDeclarator() = default;
};

}  // namespace typeart

#endif  // TYPEART_TYPEARTFUNCTIONS_H

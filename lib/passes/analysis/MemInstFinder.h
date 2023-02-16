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

#ifndef TYPEART_MEMINSTFINDER_H
#define TYPEART_MEMINSTFINDER_H

#include "MemOpData.h"

#include <memory>
#include <string>

namespace llvm {
class Module;
class Function;
class raw_ostream;
}  // namespace llvm

namespace typeart::config {
class Configuration;
}

namespace typeart::analysis {

enum class FilterImplementation { none, standard, cg, acg };

struct FunctionData {
  MallocDataList mallocs;
  FreeDataList frees;
  AllocaDataList allocas;
};

class MemInstFinder {
 public:
  // virtual void configure(MemInstFinderConfig&);
  virtual bool runOnModule(llvm::Module&)                                                = 0;
  [[nodiscard]] virtual bool hasFunctionData(const llvm::Function&) const                = 0;
  [[nodiscard]] virtual const FunctionData& getFunctionData(const llvm::Function&) const = 0;
  [[nodiscard]] virtual const GlobalDataList& getModuleGlobals() const                   = 0;
  virtual void printStats(llvm::raw_ostream&) const                                      = 0;
  virtual ~MemInstFinder()                                                               = default;
};

std::unique_ptr<MemInstFinder> create_finder(const config::Configuration&);

}  // namespace typeart::analysis

#endif  // TYPEART_MEMINSTFINDER_H

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

namespace typeart::analysis {

enum class FilterImplementation { none, standard, cg };

struct MemInstFinderConfig {
  struct Filter {
    bool ClFilterNonArrayAlloca{false};
    bool ClFilterMallocAllocPair{false};
    bool ClFilterGlobal{true};
    bool ClUseCallFilter{false};
    bool ClFilterPointerAlloca{false};

    // std::string ClCallFilterImpl{"default"};
    FilterImplementation implementation{FilterImplementation::standard};
    std::string ClCallFilterGlob{"*MPI_*"};
    std::string ClCallFilterDeepGlob{"MPI_*"};
    std::string ClCallFilterCGFile{};
  };

  bool collect_heap{false};
  bool collect_alloca{false};
  bool collect_global{false};
  Filter filter;
};

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

std::unique_ptr<MemInstFinder> create_finder(const MemInstFinderConfig&);

}  // namespace typeart::analysis

#endif  // TYPEART_MEMINSTFINDER_H

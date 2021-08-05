// TypeART library
//
// Copyright (c) 2017-2021 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#ifndef TYPEART_STANDARDFILTER_H
#define TYPEART_STANDARDFILTER_H

#include "Filter.h"
#include "support/Logger.h"
#include "support/Util.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"

#include <string>

namespace llvm {
class Argument;
class Function;
class Value;
}  // namespace llvm

namespace typeart::filter::deprecated {

using namespace llvm;

class StandardFilter final : public Filter {
  const std::string call_regex;
  bool malloc_mode{false};
  llvm::Function* start_f{nullptr};
  int depth{0};
  bool ClCallFilterDeep{true};

 public:
  explicit StandardFilter(const std::string& glob, bool CallFilterDeep);

  void setMode(bool search_malloc) override;

  void setStartingFunction(llvm::Function* start) override;

  bool filter(Value* in) override;

 private:
  bool filter(CallSite& csite, Value* in);

  bool filter(Argument* arg);

  bool shouldContinue(CallSite c, Value* in) const;

  static inline std::string getName(const Function* f);
};

}  // namespace typeart::filter::deprecated

#endif  // TYPEART_STANDARDFILTER_H

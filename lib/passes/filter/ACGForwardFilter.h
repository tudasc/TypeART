// TypeART library
//
// Copyright (c) 2017-2023 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#ifndef TYPEART_ACGFORWARDFILTER_H
#define TYPEART_ACGFORWARDFILTER_H

#include "FilterBase.h"
#include "Matcher.h"
#include "compat/CallSite.h"
#include "filter/CGInterface.h"
#include "filter/IRPath.h"
#include "MetaCG.h"
#include "MetaCGExtension.h"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/ValueMap.h>
#include <memory>
#include <string>
#include <utility>

namespace llvm {
class Function;
class Value;
}  // namespace llvm

namespace typeart::filter {
struct DefaultSearch;
}  // namespace typeart::filter

namespace typeart::filter::omp {
struct OmpContext;
}  // namespace typeart::filter::omp

namespace typeart::filter {


struct FunctionSignature {
  /**
   * the identifier/name of the function
   */
  const std::string identifier{"*"};
  /**
   * the types of the formal arguments
   */
  const std::vector<std::string> paramTypes{};
  /**
   * the type of the return value
   */
  const std::string returnType{};
  /**
   * indicates that this function may accept an arbitrary number of formal arguments
   */
  const bool isVariadic = false;

  [[nodiscard]] bool paramIsType(unsigned ArgumentNumber, const std::string& Type) const noexcept {
    if (ArgumentNumber >= paramTypes.size()) {
      return isVariadic;
    }
    return paramTypes[ArgumentNumber] == Type;
  }

  [[nodiscard]] bool returnIsType(const std::string& Type) const noexcept {
    return returnType == Type;
  }
};

// ipdf fulfills 2 different tasks:
//  1) possible callees based on the annotated callsite-id
//  2) flow-through arguments, used for argument-based flow calculation
struct FunctionDescriptor {
  struct ArgumentEdge {
    /// The position of the formal argument of the callee.
    const int argumentNumber;

    /// reference to the callee
    const FunctionDescriptor& callee;
  };

  /// conservatively assume a function is a target unless defined otherwise
  bool isTarget = true;

  /// assume a function has no definition unless defined otherwise
  bool isDefinition = false;

  /// the key represents the actual argument position of the caller function (this function)
  std::multimap<int, const ArgumentEdge> reachableFunctionArguments{};

  /// maps a callsite-id to its callees
  std::multimap<uint64_t, const FunctionDescriptor*> callsiteCallees{};

  /// signature of the function
  FunctionSignature functionSignature;
};

using ACGDataMap = llvm::StringMap<FunctionDescriptor>;

using JSONACG = metacg::MetaCG<metacg::FunctionSignature, metacg::InterDataFlow>;

/// converts the JSON structure in a better processable structure
ACGDataMap createDatabase(const Regex&, JSONACG&);

struct ACGFilterTrait {
  constexpr static bool Indirect    = true;
  constexpr static bool Intrinsic   = false;
  constexpr static bool Declaration = true;
  constexpr static bool Definition  = true;
  constexpr static bool PreCheck    = true;
};

class ACGFilterImpl {
 public:
  using Support = ACGFilterTrait;

  explicit ACGFilterImpl(ACGDataMap&& dataMap) : functionMap(std::move(dataMap)) {
  }

  [[nodiscard]] FilterAnalysis precheck(llvm::Value*, llvm::Function*, const FPath&);

  [[nodiscard]] FilterAnalysis decl(const llvm::CallSite&, const Path&);

  [[nodiscard]] FilterAnalysis def(const llvm::CallSite&, const Path&);

  [[nodiscard]] FilterAnalysis indirect(const llvm::CallSite&, const Path&);

 private:
  using functionmap_t   = llvm::ValueMap<const llvm::Function*, unsigned>;
  using identifiermap_t = llvm::ValueMap<const llvm::Instruction*, unsigned>;

  FunctionOracleMatcher candidateMatcher{};
  ACGDataMap functionMap;
  functionmap_t analyzedFunctions{};
  identifiermap_t callSiteIdentifiers{};

  [[nodiscard]] FilterAnalysis analyseFlowPath(const std::vector<FunctionDescriptor::ArgumentEdge>&) const;

  template <typename RangeT>
  [[nodiscard]] FilterAnalysis analyseMaybeCandidates(const RangeT&&) const;

  [[nodiscard]] FilterAnalysis analyseCallsite(const llvm::CallBase&, const Path&) const;

  [[nodiscard]] unsigned int getIdentifierForCallsite(const llvm::CallBase&) const;

  [[nodiscard]] std::vector<const FunctionDescriptor*> getCalleesForCallsite(const FunctionDescriptor&,
                                                                             const llvm::CallBase&) const;

  unsigned calculateSiteIdentifiersIfAbsent(const llvm::Function&);
};

using ACGForwardFilter = BaseFilter<ACGFilterImpl, DefaultSearch, omp::OmpContext>;

}  // namespace typeart::filter

#endif  // TYPEART_ACGFORWARDFILTER_H
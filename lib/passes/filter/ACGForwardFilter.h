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
#include "support/MetaCG.h"
#include "support/MetaCGExtension.h"

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

using namespace util::metacg;

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

  [[nodiscard]] constexpr bool paramIsType(size_t No, const std::string& Type) const noexcept {
    if (No >= paramTypes.size()) {
      return isVariadic;
    }
    return paramTypes[No] == Type;
  }

  [[nodiscard]] constexpr bool returnIsType(const std::string& Type) const noexcept {
    return returnType == Type;
  }
};

// ipdf fulfills 2 different tasks:
//  1) possible callees based on the annotated callsite-id
//  2) flow-through arguments, used for argument-based flow calculation
struct FunctionDescriptor {
  struct ArgumentEdge {
    /// The position of the formal argument of the callee.
    const int formalArgumentNumber;

    /// reference to the callee
    const FunctionDescriptor& callee;
  };

  /// conservatively assume a function is a target unless defined otherwise
  bool isTarget = true;

  /// assume a function has no definition unless defined otherwise
  bool isDefinition = false;

  /// the key represents the actual argument position of the caller function (this function)
  std::multimap<int, const ArgumentEdge> reachableFormals{};

  /// maps a callsite-id to its callees
  std::multimap<uint64_t, const FunctionDescriptor*> callsiteCallees{};

  /// signature of the function
  FunctionSignature functionSignature;
};

using ACGDataMap = llvm::StringMap<FunctionDescriptor>;

using JSONACG = MetaCG<Signature, InterDataFlow>;

/// converts the JSON structure in a better processable structure
ACGDataMap createDatabase(const Regex& MetaCGNode, JSONACG& MetaCg);

class ACGFilterImpl {
 private:
  using functionmap_t   = llvm::ValueMap<const llvm::Function*, unsigned>;
  using identifiermap_t = llvm::ValueMap<const llvm::Instruction*, unsigned>;

  static constexpr auto VoidType = "i8*";

  AnyMatcher candidateMatcher{};
  ACGDataMap functionMap;
  functionmap_t analysedFunctions{};
  identifiermap_t callSiteIdentifiers{};

  [[nodiscard]] bool edgeReachesRelevantFormalArgument(const FunctionDescriptor::ArgumentEdge&) const;
  [[nodiscard]] bool edgeMaybeReachesFormalArgument(const FunctionDescriptor::ArgumentEdge&) const;

  [[nodiscard]] std::vector<FunctionDescriptor::ArgumentEdge> createEdgesForCallsite(
      const CallBase&, const llvm::Value&, const std::vector<const FunctionDescriptor*>&) const;

  [[nodiscard]] FilterAnalysis analyseFlowPath(std::vector<FunctionDescriptor::ArgumentEdge>&& InitialEdges) const;

  template <typename RangeT>
  [[nodiscard]] FilterAnalysis analyseMaybeCandidates(const RangeT&&) const;

  [[nodiscard]] FilterAnalysis analyseCallsite(const llvm::CallBase&, const Path&) const;

  [[nodiscard]] unsigned int getIdentifierForCallsite(const llvm::CallBase& Site) const;

  [[nodiscard]] bool isFormalArgumentRelevant(const FunctionDescriptor::ArgumentEdge&) const;

  [[nodiscard]] std::vector<const FunctionDescriptor*> getCalleesForCallsite(const FunctionDescriptor& FunctionData,
                                                                             const CallBase& Site) const;

  std::size_t calculateSiteIdentifiersIfAbsent(const Function& Function);

 public:
  explicit ACGFilterImpl(ACGDataMap&& DataMap) : functionMap(std::move(DataMap)) {
  }

  [[nodiscard]] FilterAnalysis precheck(llvm::Value* in, llvm::Function*, const FPath&);

  [[nodiscard]] FilterAnalysis decl(const llvm::CallSite&, const Path&);

  [[nodiscard]] FilterAnalysis def(const llvm::CallSite&, const Path&);

  [[nodiscard]] FilterAnalysis indirect(const llvm::CallSite&, const Path&);
};

using ACGForwardFilter = BaseFilter<ACGFilterImpl, DefaultSearch, omp::OmpContext>;

}  // namespace typeart::filter

#endif  // TYPEART_ACGFORWARDFILTER_H
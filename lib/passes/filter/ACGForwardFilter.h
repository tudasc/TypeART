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
#include "MetaCG.h"
#include "MetaCGExtension.h"
#include "compat/CallSite.h"
#include "filter/CGInterface.h"
#include "filter/IRPath.h"

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
  const std::vector<std::string> param_types{};
  /**
   * the type of the return value
   */
  const std::string return_type{};
  /**
   * indicates that this function may accept an arbitrary number of formal arguments
   */
  const bool is_variadic = false;

  template <typename TypeID>
  [[nodiscard]] inline bool paramIsType(unsigned argument_number, TypeID&& is_type) const noexcept {
    if (argument_number >= param_types.size()) {
      return is_variadic;
    }
    return is_type(param_types[argument_number]);
  }

  template <typename TypeID>
  [[nodiscard]] inline bool returnIsType(TypeID&& is_type) const noexcept {
    return is_type(return_type);
  }
};

// ipdf fulfills two different tasks:
//  1) possible callees based on the annotated callsite-id.
//     these are used to determine which functions can be reached from a given callsite.
//
//  2) function argument based inter-procedural dataflow.
//     this is used to model which other function arguments can be reached (the sink-arguments)
//     from a given function argument (the source-argument)
struct FunctionDescriptor {
  struct ArgumentEdge {
    /// The position of the (sink) argument of the callee.
    const int argument_number;

    /// reference to the callee
    const FunctionDescriptor& callee;
  };

  /// conservatively assume a function is a target unless defined otherwise
  bool is_target = true;

  /// assume a function has no definition unless defined otherwise
  bool is_definition = false;

  /// the key represents the source argument position of the caller function (this function)
  /// the values represent reachable functions (with the corresponding argument number)
  std::multimap<int, const ArgumentEdge> reachable_function_arguments{};

  /// maps a callsite-id to its callees
  std::multimap<uint64_t, const FunctionDescriptor*> callsite_callees{};

  /// signature of the function
  FunctionSignature function_signature;
};

using ACGDataMap = llvm::StringMap<FunctionDescriptor>;

using JsonACG = metacg::MetaCG<metacg::MetaFieldGroup<metacg::FunctionSignature, metacg::InterDataFlow>>;

/// converts the JSON structure in a better processable structure
ACGDataMap createDatabase(const Regex&, JsonACG&);

/// calculates the identifier for a function node in the acg
class FunctionIdentification {
 public:
  std::string getIdentifierForFunction(const llvm::Function& function) const;
};

class CallSiteIdentification {
  /// stores the number of callsites within a functions
  std::map<const llvm::Function*, unsigned> analyzedFunctions_{};
  // stores the identifier of a callsite
  std::map<const llvm::CallBase*, unsigned> callSiteIdentifiers_{};

 public:
  /// identifies all callsites of a function and returns the number of callsites of the given function
  unsigned calculateCallsiteIdentifiersIfAbsent(const llvm::Function&);
  /// returns the identifier for a call-site
  [[nodiscard]] unsigned getIdentifierForCallsite(const llvm::CallBase&) const;
};

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

  explicit ACGFilterImpl(ACGDataMap&& data_map) : functionMap_(std::move(data_map)) {
  }

  [[nodiscard]] FilterAnalysis precheck(llvm::Value*, llvm::Function*, const FPath&);

  [[nodiscard]] FilterAnalysis decl(const llvm::CallSite&, const Path&);

  [[nodiscard]] FilterAnalysis def(const llvm::CallSite&, const Path&);

  [[nodiscard]] FilterAnalysis indirect(const llvm::CallSite&, const Path&);

 private:
  FunctionIdentification functionIdentification_{};
  CallSiteIdentification callSiteIdentification_{};
  FunctionOracleMatcher candidateMatcher_{};
  ACGDataMap functionMap_;

  [[nodiscard]] FilterAnalysis analyseFlowPath(const std::vector<FunctionDescriptor::ArgumentEdge>&) const;

  template <typename RangeT>
  [[nodiscard]] FilterAnalysis analyseMaybeCandidates(const RangeT&&) const;

  [[nodiscard]] FilterAnalysis analyseCallsite(const llvm::CallBase&, const Path&) const;

  [[nodiscard]] std::vector<const FunctionDescriptor*> getCalleesForCallsite(const FunctionDescriptor&,
                                                                             const llvm::CallBase&) const;
};

using ACGForwardFilter = BaseFilter<ACGFilterImpl, DefaultSearch, omp::OmpContext>;

}  // namespace typeart::filter

#endif  // TYPEART_ACGFORWARDFILTER_H

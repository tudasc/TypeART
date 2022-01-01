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

#include "CGForwardFilter.h"

#include "CGInterface.h"
#include "Matcher.h"
#include "OmpUtil.h"
#include "filter/FilterBase.h"
#include "filter/FilterUtil.h"
#include "support/Logger.h"
#include "support/Util.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <utility>

namespace llvm {
class Function;
}  // namespace llvm

namespace typeart::filter {

CGFilterImpl::CGFilterImpl(const std::string& filter_str, std::unique_ptr<CGInterface>&& cgraph)
    : CGFilterImpl(filter_str, std::move(cgraph), nullptr) {
}

CGFilterImpl::CGFilterImpl(const std::string& filter_str, std::unique_ptr<CGInterface>&& cgraph,
                           std::unique_ptr<Matcher>&& matcher)
    : filter(util::glob2regex(filter_str)), call_graph(std::move(cgraph)), deep_matcher(std::move(matcher)) {
}

FilterAnalysis CGFilterImpl::precheck(Value* in, Function* start, const FPath& fpath) {
  if (start == nullptr) {
    return FilterAnalysis::Continue;
  }

  FunctionAnalysis analysis;
  analysis.analyze(start);
  if (analysis.empty()) {
    return FilterAnalysis::Filter;
  }

  if (fpath.empty()) {
    // These conditions (temp alloc and alloca reaches task)
    // are only interesting if filter just started (aka fpath is empty)
    if (isTempAlloc(in)) {
      LOG_DEBUG("Alloca is a temporary " << *in);
      return FilterAnalysis::Filter;
    }

    if (llvm::AllocaInst* alloc = llvm::dyn_cast<AllocaInst>(in)) {
      if (alloc->getAllocatedType()->isStructTy() && omp::OmpContext::allocaReachesTask(alloc)) {
        LOG_DEBUG("Alloca reaches task call " << *alloc)
        return FilterAnalysis::Filter;
      }
    }
  }

  const auto has_omp_task =
      llvm::any_of(analysis.calls.decl, [](const auto& csite) { return omp::OmpContext::isOmpTaskRelated(csite); });
  if (has_omp_task) {
    // FIXME we cannot handle complex data flow of tasks at this point, hence, this check
    LOG_DEBUG("Keep value " << *in << ". Detected omp task call.");
    return FilterAnalysis::Keep;
  }

  return FilterAnalysis::Continue;
}

FilterAnalysis CGFilterImpl::decl(CallSite current, const Path& p) {
  if (deep_matcher && deep_matcher->match(current) == Matcher::MatchResult::Match) {
    auto result = correlate2void(current, p);
    switch (result) {
      case ArgCorrelation::GlobalMismatch:
        [[fallthrough]];
      case ArgCorrelation::ExactMismatch:
        LOG_DEBUG("Correlated, continue search");
        return FilterAnalysis::Continue;
      default:
        return FilterAnalysis::Keep;
    }
  }

  const auto searchCG = [&](auto from) {
    if (call_graph) {
      return call_graph->reachable(std::string{from->getName()}, filter);
    }
    return CGInterface::ReachabilityResult::unknown;
  };

  const auto reached = searchCG(current.getCalledFunction());

  switch (reached) {
    case CGInterface::ReachabilityResult::reaches:
      return FilterAnalysis::Keep;
    case CGInterface::ReachabilityResult::never_reaches:
      return FilterAnalysis::Skip;
    case CGInterface::ReachabilityResult::maybe_reaches:
      return FilterAnalysis::Filter;
    default:
      return FilterAnalysis::Continue;
  }
}

FilterAnalysis CGFilterImpl::def(CallSite current, const Path& p) {
  return decl(current, p);
}

}  // namespace typeart::filter
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

#include "StdForwardFilter.h"

#include "OmpUtil.h"
#include "filter/FilterBase.h"
#include "filter/FilterUtil.h"
#include "filter/Matcher.h"
#include "support/Logger.h"

#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <utility>

namespace llvm {
class Function;
}  // namespace llvm

namespace typeart::filter {

ForwardFilterImpl::ForwardFilterImpl(std::unique_ptr<Matcher>&& m) : ForwardFilterImpl(std::move(m), nullptr) {
}

ForwardFilterImpl::ForwardFilterImpl(std::unique_ptr<Matcher>&& m, std::unique_ptr<Matcher>&& deep)
    : matcher(std::move(m)), deep_matcher(std::move(deep)) {
}

FilterAnalysis filter::ForwardFilterImpl::precheck(Value* in, Function* start, const FPath& fpath) {
  if (start == nullptr) {
    // In case of global var.
    return FilterAnalysis::Continue;
  }

  FunctionAnalysis analysis;
  analysis.analyze(start);
  if (analysis.empty()) {
    return FilterAnalysis::Filter;
  }

  if (!fpath.empty()) {
    // the value is part of a call chain
    return FilterAnalysis::Continue;
  }

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

  return FilterAnalysis::Continue;
}

FilterAnalysis filter::ForwardFilterImpl::decl(CallSite current, const Path& p) const {
  const bool match_sig = matcher->match(current) == Matcher::MatchResult::Match;
  if (match_sig) {
    // if we have a deep_matcher it needs to trigger, otherwise analyze
    if (!deep_matcher || deep_matcher->match(current) == Matcher::MatchResult::Match) {
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
  }
  // Not a relevant name (e.g. MPI), ask oracle if we have
  // some benign (C) function name
  const auto oracle_match = oracle.match(current);
  switch (oracle_match) {
    case Matcher::MatchResult::ShouldSkip: {
      return FilterAnalysis::Skip;
    }
    case Matcher::MatchResult::ShouldContinue: {
      return FilterAnalysis::Continue;
    }
    default:
      break;
  }

  return FilterAnalysis::Keep;
}

FilterAnalysis filter::ForwardFilterImpl::def(CallSite current, const Path& p) const {
  const bool match_sig = matcher->match(current) == Matcher::MatchResult::Match;
  if (match_sig) {
    if (!deep_matcher || deep_matcher->match(current) == Matcher::MatchResult::Match) {
      auto result = correlate2void(current, p);
      switch (result) {
        case ArgCorrelation::GlobalMismatch:
          [[fallthrough]];
        case ArgCorrelation::ExactMismatch:
          LOG_DEBUG("Correlated definition args, continue search");
          return FilterAnalysis::Continue;
        default:
          return FilterAnalysis::Keep;
      }
    } else {
      return FilterAnalysis::Keep;
    }
  }

  return FilterAnalysis::FollowDef;
}

}  // namespace typeart::filter
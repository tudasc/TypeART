//
// Created by ahueck on 26.10.20.
//

#include "StdForwardFilter.h"

#include "OmpUtil.h"

namespace typeart::filter {

ForwardFilterImpl::ForwardFilterImpl(std::unique_ptr<Matcher>&& m) : ForwardFilterImpl(std::move(m), nullptr) {
}

ForwardFilterImpl::ForwardFilterImpl(std::unique_ptr<Matcher>&& m, std::unique_ptr<Matcher>&& deep)
    : matcher(std::move(m)), deep_matcher(std::move(deep)) {
}

FilterAnalysis filter::ForwardFilterImpl::precheck(Value* in, Function* start, const FPath& fpath) {
  if (start != nullptr) {
    FunctionAnalysis analysis;
    analysis.analyze(start);
    if (analysis.empty()) {
      return FilterAnalysis::Filter;
    }

    if (fpath.empty()) {
      auto temp = isTempAlloc(in);
      if (temp) {
        LOG_DEBUG("Alloca is a temporary " << *in);
        return FilterAnalysis::Filter;
      }
      if (llvm::AllocaInst* alloc = llvm::dyn_cast<AllocaInst>(in); !temp && alloc != nullptr) {
        if (alloc->getAllocatedType()->isStructTy()) {
          const bool reaches = omp::OmpContext::allocaReachesTask(alloc);
          if (reaches) {
            LOG_DEBUG("Alloca reaches task call " << *alloc)
            return FilterAnalysis::Filter;
          }
        }
      }
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
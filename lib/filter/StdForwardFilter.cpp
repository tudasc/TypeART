//
// Created by ahueck on 26.10.20.
//

#include "StdForwardFilter.h"

namespace typeart::filter {

filter::ForwardFilterImpl::ForwardFilterImpl(std::string filter, bool deep)
    : filter(util::glob2regex(std::move(filter))), deep(deep) {
}

FilterAnalysis filter::ForwardFilterImpl::precheck(Value* in, Function* start) {
  if (start) {
    FunctionAnalysis analysis;
    analysis.analyze(start);
    if (analysis.empty()) {
      return FilterAnalysis::Filter;
    }
  }
  return FilterAnalysis::Continue;
}

FilterAnalysis filter::ForwardFilterImpl::decl(CallSite current, const Path& p) {
  auto call_target    = current.getCalledFunction();
  const bool matchSig = match(call_target);
  if (matchSig && deep) {
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

  return FilterAnalysis::Keep;
}

FilterAnalysis filter::ForwardFilterImpl::def(CallSite current, const Path& p) {
  auto call_target     = current.getCalledFunction();
  const bool match_sig = match(call_target);
  if (match_sig) {
    if (deep) {
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

bool filter::ForwardFilterImpl::match(Function* callee) {
  const auto f_name = util::demangle(callee->getName());
  const auto result = util::regex_matches(filter, f_name);
  LOG_DEBUG("Matching " << f_name << " against " << filter << " : " << result)
  return result;
}

}  // namespace typeart::filter
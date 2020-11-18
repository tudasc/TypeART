//
// Created by ahueck on 26.10.20.
//

#include "CGForwardFilter.h"

#include "CGInterface.h"
#include "Matcher.h"

namespace typeart::filter {

CGFilterImpl::CGFilterImpl(std::string fstr, std::string cgFile)
    : filter(util::glob2regex(std::move(fstr))), matcher(filter) {
  if (!callGraph && !cgFile.empty()) {
    LOG_DEBUG("Resetting the CGInterface with JSON CG");
    callGraph.reset(JSONCG::getJSON(cgFile));
  } else {
    LOG_FATAL("CG File not found " << cgFile);
  }
}

FilterAnalysis CGFilterImpl::precheck(Value* in, Function* start) {
  if (start) {
    FunctionAnalysis analysis;
    analysis.analyze(start);
    if (analysis.empty()) {
      return FilterAnalysis::Filter;
    }
  }
  return FilterAnalysis::Continue;
}

FilterAnalysis CGFilterImpl::decl(CallSite current, const Path& p) {
  // deeper analysis only possible if we had a path from *in* to *current*
  const bool matchSig = matcher.match(current);
  if (matchSig) {
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
    if (callGraph) {
      return callGraph->reachable(from->getName(), filter);
    } else {
      return CGInterface::ReachabilityResult::unknown;
    }
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
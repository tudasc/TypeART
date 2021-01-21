//
// Created by ahueck on 26.10.20.
//

#include "CGForwardFilter.h"

#include "CGInterface.h"
#include "Matcher.h"
#include "OmpUtil.h"

namespace typeart::filter {

CGFilterImpl::CGFilterImpl(const std::string& filter_str, std::unique_ptr<CGInterface>&& cgraph)
    : CGFilterImpl(filter_str, std::move(cgraph), nullptr) {
}

CGFilterImpl::CGFilterImpl(const std::string& filter_str, std::unique_ptr<CGInterface>&& cgraph,
                           std::unique_ptr<Matcher>&& matcher)
    : filter(util::glob2regex(filter_str)), call_graph(std::move(cgraph)), deep_matcher(std::move(matcher)) {
}

FilterAnalysis CGFilterImpl::precheck(Value* in, Function* start, const FPath&) {
  if (start != nullptr) {
    FunctionAnalysis analysis;
    analysis.analyze(start);
    if (analysis.empty()) {
      return FilterAnalysis::Filter;
    }

    const auto has_omp_task = [](const auto& callsites_v) {
      for (const auto& csite : callsites_v) {
        if (omp::OmpContext::isOmpTaskRelated(csite)) {
          return true;
        }
      }
      return false;
    };
    if (has_omp_task(analysis.calls.decl)) {
      // FIXME we cannot handle complex data flow of tasks at this point, hence, this check
      LOG_DEBUG("Keep value " << *in << ". Detected omp task call.");
      return FilterAnalysis::Keep;
    }
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
      return call_graph->reachable(from->getName(), filter);
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
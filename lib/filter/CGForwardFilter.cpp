//
// Created by ahueck on 26.10.20.
//

#include "CGForwardFilter.h"

#include "CGInterface.h"

namespace typeart::filter {

CGFilterImpl::CGFilterImpl(std::string filter, std::string cgFile) : filter(util::glob2regex(std::move(filter))) {
  if (!callGraph && !cgFile.empty()) {
    LOG_DEBUG("Resetting the CGInterface with JSON CG");
    callGraph.reset(JSONCG::getJSON(cgFile));
  } else {
    LOG_FATAL("CG File not found " << cgFile);
  }
}

FilterAnalysis CGFilterImpl::precheck(Value* in, Function* start) {
  return FilterAnalysis::cont;
}

FilterAnalysis CGFilterImpl::decl(Value* in, CallSite current) {
  // deeper analysis only possible if we had a path from *in* to *current*
  const bool matchSig = match(current.getCalledFunction());
  if (matchSig) {
    return FilterAnalysis::keep;
  }

  const auto searchCG = [&](auto from) {
    if (callGraph) {
      return callGraph->reachable(from->getName(), filter);
    } else {
      return CGInterface::ReachabilityResult::unknown;
    }
  };

  const auto reached = searchCG(current.getCalledFunction());

  if (reached == CGInterface::ReachabilityResult::reaches) {
    return FilterAnalysis::keep;
  } else if (reached == CGInterface::ReachabilityResult::never_reaches) {
    return FilterAnalysis::skip;
  } else if (reached == CGInterface::ReachabilityResult::maybe_reaches) {
    return FilterAnalysis::filter;  // XXX This should be where we can change semantics
  }

  return FilterAnalysis::cont;
}

FilterAnalysis CGFilterImpl::def(Value* in, CallSite current) {
  return decl(in, current);
}

bool CGFilterImpl::match(Function* callee) {
  const auto f_name = util::demangle(callee->getName());
  return util::regex_matches(filter, f_name);
}

}  // namespace typeart::filter
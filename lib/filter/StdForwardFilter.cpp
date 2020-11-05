//
// Created by ahueck on 26.10.20.
//

#include "StdForwardFilter.h"

#include "../support/TypeUtil.h"

namespace typeart::filter {

filter::Handler::Handler(std::string filter) : filter(util::glob2regex(std::move(filter))) {
}

FilterAnalysis filter::Handler::precheck(Value* in, Function* start) {
  if (start) {
    FunctionAnalysis analysis;
    analysis.analyze(start);
    if (analysis.empty()) {
      return FilterAnalysis::filter;
    }
  }
  return FilterAnalysis::cont;
}

FilterAnalysis filter::Handler::decl(CallSite current, const Path& p) {
  const bool matchSig = match(current.getCalledFunction());
  if (matchSig) {
    auto result = correlate2void(current, p);
    switch (result) {
      case ArgCorrelation::GlobalMismatch:
        [[fallthrough]];
      case ArgCorrelation::ExactMismatch:
        LOG_DEBUG("Correlated, continue search");
        return FilterAnalysis::cont;
      default:
        return FilterAnalysis::keep;
    }
  }
  return FilterAnalysis::keep;
}

FilterAnalysis filter::Handler::def(CallSite current, const Path& p) {
  auto callTarget = current.getCalledFunction();

  if (match(callTarget)) {
    auto result = correlate2void(current, p);
    switch (result) {
      case ArgCorrelation::GlobalMismatch:
        [[fallthrough]];
      case ArgCorrelation::ExactMismatch:
        LOG_DEBUG("Correlated definition args, continue search");
        return FilterAnalysis::cont;
      default:
        return FilterAnalysis::keep;
    }
  }

  return FilterAnalysis::follow;
}

bool filter::Handler::match(Function* callee) {
  const auto f_name = util::demangle(callee->getName());
  return util::regex_matches(filter, f_name);
}

}  // namespace typeart::filter
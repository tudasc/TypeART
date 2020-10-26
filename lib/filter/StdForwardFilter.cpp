//
// Created by ahueck on 26.10.20.
//

#include "StdForwardFilter.h"

namespace typeart::filter {

filter::Handler::Handler(std::string filter) : filter(util::glob2regex(std::move(filter))) {
}

FilterAnalysis filter::Handler::precheck(Value* in, Function* start) {
  if (start) {
    FunctionAnalysis analysis;
    analysis.analyze(start);
    if (analysis.empty()) {
      return FilterAnalysis::skip;
    }
  }
  return FilterAnalysis::filter;
}

FilterAnalysis filter::Handler::decl(Value* in, CallSite current) {
  // deeper analysis only possible if we had a path from *in* to *current*
  const bool matchSig = match(current.getCalledFunction());
  if (matchSig) {
    return FilterAnalysis::keep;
  }
  return FilterAnalysis::skip;
}

FilterAnalysis filter::Handler::def(Value* in, CallSite current) {
  // scan only first level, TODO recurse all:
  FunctionAnalysis analysis;
  analysis.analyze(current.getCalledFunction());
  if (analysis.empty()) {
    return FilterAnalysis::skip;
  }
  return FilterAnalysis::keep;
}

bool filter::Handler::match(Function* callee) {
  const auto f_name = util::demangle(callee->getName());
  return util::regex_matches(filter, f_name);
}

}  // namespace typeart::filter
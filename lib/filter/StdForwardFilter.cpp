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
      return FilterAnalysis::filter;
    }
  }
  return FilterAnalysis::cont;
}

FilterAnalysis filter::Handler::decl(Value* in, CallSite current) {
  // deeper analysis only possible if we had a path from *in* to *current*
  const bool matchSig = match(current.getCalledFunction());
  if (matchSig) {
    return FilterAnalysis::keep;
  }
  return FilterAnalysis::keep;
}

FilterAnalysis filter::Handler::def(Value* in, CallSite current) {
  // scan only first level, TODO recurse all:
  auto callTarget = current.getCalledFunction();

  if (match(callTarget)) {
    return FilterAnalysis::keep;
  }

  // in case of recursive call ...
  if (auto* inst = llvm::dyn_cast<Instruction>(in)) {
    auto parentF = inst->getFunction();
    if (parentF == callTarget) {
      return FilterAnalysis::skip;
    }
  }

  FunctionAnalysis analysis;
  analysis.analyze(callTarget);
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
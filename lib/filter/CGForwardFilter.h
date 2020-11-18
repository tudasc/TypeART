//
// Created by ahueck on 26.10.20.
//

#ifndef TYPEART_CGFORWARDFILTER_H
#define TYPEART_CGFORWARDFILTER_H

#include "FilterBase.h"
#include "Matcher.h"

namespace typeart::filter {

struct CGFilterTrait {
  constexpr static bool Indirect    = false;
  constexpr static bool Intrinsic   = false;
  constexpr static bool Declaration = true;
  constexpr static bool Definition  = true;
  constexpr static bool PreCheck    = true;
};

class CGInterface;

struct CGFilterImpl {
  using Support = CGFilterTrait;

  std::string filter;
  std::unique_ptr<CGInterface> callGraph;
  DefaultStringMatcher matcher;

  CGFilterImpl(std::string filter, std::string cgFile);

  FilterAnalysis precheck(Value* in, Function* start);

  FilterAnalysis decl(CallSite current, const Path& p);

  FilterAnalysis def(CallSite current, const Path& p);
};

using CGForwardFilter = BaseFilter<CGFilterImpl, DefaultSearch>;

}  // namespace typeart::filter

#endif  // TYPEART_CGFORWARDFILTER_H
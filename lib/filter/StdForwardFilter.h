//
// Created by ahueck on 26.10.20.
//

#ifndef TYPEART_STDFORWARDFILTER_H
#define TYPEART_STDFORWARDFILTER_H

#include "FilterBase.h"

namespace typeart::filter {

struct FilterTrait {
  constexpr static bool Indirect    = false;
  constexpr static bool Intrinsic   = false;
  constexpr static bool Declaration = true;
  constexpr static bool Definition  = true;
  constexpr static bool PreCheck    = true;
};

struct Handler {
  using Support = FilterTrait;

  std::string filter;

  Handler(std::string filter);

  FilterAnalysis precheck(Value* in, Function* start);

  FilterAnalysis decl(CallSite current, const Path& p);

  FilterAnalysis def(CallSite current, const Path& p);

 private:
  bool match(Function* callee);
};

using StandardForwardFilter = BaseFilter<Handler, SearchStoreDir>;

}  // namespace typeart::filter

#endif  // TYPEART_STDFORWARDFILTER_H

//
// Created by ahueck on 26.10.20.
//

#ifndef TYPEART_CGFORWARDFILTER_H
#define TYPEART_CGFORWARDFILTER_H

#include "FilterBase.h"

namespace typeart::filter {

struct FilterTraitCG {
  constexpr static bool Indirect    = false;
  constexpr static bool Intrinsic   = false;
  constexpr static bool Declaration = true;
  constexpr static bool Definition  = true;
  constexpr static bool PreCheck    = false;  // TODO impl and switch
};

class CGInterface;

struct CGFilterImpl {
  using Support = FilterTraitCG;

  std::string filter;
  std::unique_ptr<CGInterface> callGraph;

  CGFilterImpl(std::string filter, std::string cgFile);

  FilterAnalysis precheck(Value* in, Function* start);

  FilterAnalysis decl(CallSite current, const Path& p);

  FilterAnalysis def(CallSite current, const Path& p);

 private:
  bool match(Function* callee);
};

using CGForwardFilter = BaseFilter<CGFilterImpl, SearchStoreDir>;

}  // namespace typeart::filter

#endif  // TYPEART_CGFORWARDFILTER_H
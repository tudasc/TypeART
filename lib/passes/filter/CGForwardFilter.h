// TypeART library
//
// Copyright (c) 2017-2026 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#ifndef TYPEART_CGFORWARDFILTER_H
#define TYPEART_CGFORWARDFILTER_H

#include "FilterBase.h"
#include "Matcher.h"
#include "MetaCG.h"
#include "compat/CallSite.h"

namespace typeart::filter {

namespace omp {
struct OmpContext;
}

struct DefaultSearch;

struct CGForwardFilterTrait {
  constexpr static bool Indirect    = true;
  constexpr static bool Intrinsic   = false;
  constexpr static bool Declaration = true;
  constexpr static bool Definition  = true;
  constexpr static bool PreCheck    = true;
};

struct CGForwardFilterImpl {
  using Support = CGForwardFilterTrait;

  CGForwardFilterImpl(metacg::Mcg&&, std::unique_ptr<Matcher>&& m, std::unique_ptr<Matcher>&& deep);

  FilterAnalysis precheck(Value*, Function*, const FPath&);
  FilterAnalysis indirect(CallSite, const Path&);
  FilterAnalysis decl(CallSite, const Path&);
  FilterAnalysis def(CallSite, const Path&);

 private:
  FilterAnalysis reachesMatching(ArrayRef<size_t>);

  metacg::Mcg mcg;
  // Regex matcher;
  std::unique_ptr<Matcher> matcher;
  std::unique_ptr<Matcher> deep_matcher;
  FunctionOracleMatcher oracle;
};

using CGForwardFilter = BaseFilter<CGForwardFilterImpl, DefaultSearch, omp::OmpContext>;

}  // namespace typeart::filter

#endif  // TYPEART_CGFORWARDFILTER_H

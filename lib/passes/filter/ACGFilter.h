// TypeART library
//
// Copyright (c) 2017-2023 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#ifndef ARGFLOWFILTER_H
#define ARGFLOWFILTER_H

#include "compat/CallSite.h"
#include "FilterBase.h"
#include "MetaCG.h"

namespace typeart::filter {

namespace omp {
  struct OmpContext;
}

struct DefaultSearch;

struct ArgflowFilterTrait {
  constexpr static bool Indirect    = false;
  constexpr static bool Intrinsic   = false;
  constexpr static bool Declaration = true;
  constexpr static bool Definition  = true;
  constexpr static bool PreCheck    = true;
};

class CGInterface;

struct AcgFilterImpl {
  using Support = ArgflowFilterTrait;

  AcgFilterImpl(metacg::Mcg&&, Regex&&);

  FilterAnalysis precheck(Value*, Function*, const FPath&);
  FilterAnalysis decl(CallSite, const Path&);
  FilterAnalysis def(CallSite, const Path&);

private:
  FilterAnalysis reachesMatching(ArrayRef<size_t>, size_t);

  metacg::Mcg mcg;
  Regex matcher;
};

using AcgFilter = BaseFilter<AcgFilterImpl, DefaultSearch, omp::OmpContext>;

} // namespace typeart::filter

#endif //ARGFLOWFILTER_H

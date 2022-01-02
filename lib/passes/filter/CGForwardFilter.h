// TypeART library
//
// Copyright (c) 2017-2022 TypeART Authors
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
#include "compat/CallSite.h"
#include "filter/CGInterface.h"
#include "filter/IRPath.h"

#include <memory>
#include <string>

namespace llvm {
class Function;
class Value;
}  // namespace llvm
namespace typeart {
namespace filter {
namespace omp {
struct OmpContext;
}  // namespace omp
struct DefaultSearch;
}  // namespace filter
}  // namespace typeart

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
  std::unique_ptr<CGInterface> call_graph;
  std::unique_ptr<Matcher> deep_matcher;

  CGFilterImpl(const std::string& filter_str, std::unique_ptr<CGInterface>&& cgraph);

  CGFilterImpl(const std::string& filter_str, std::unique_ptr<CGInterface>&& cgraph,
               std::unique_ptr<Matcher>&& matcher);

  FilterAnalysis precheck(Value* in, Function* start, const FPath&);

  FilterAnalysis decl(CallSite current, const Path& p);

  FilterAnalysis def(CallSite current, const Path& p);
};

using CGForwardFilter = BaseFilter<CGFilterImpl, DefaultSearch, omp::OmpContext>;

}  // namespace typeart::filter

#endif  // TYPEART_CGFORWARDFILTER_H
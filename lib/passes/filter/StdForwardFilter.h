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

#ifndef TYPEART_STDFORWARDFILTER_H
#define TYPEART_STDFORWARDFILTER_H

#include "FilterBase.h"
#include "Matcher.h"
#include "compat/CallSite.h"
#include "filter/IRPath.h"

#include <memory>

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

class ForwardFilterImpl {
 private:
  std::unique_ptr<Matcher> matcher;
  std::unique_ptr<Matcher> deep_matcher;
  FunctionOracleMatcher oracle;  // TODO make set flexible

 public:
  explicit ForwardFilterImpl(std::unique_ptr<Matcher>&& m);

  ForwardFilterImpl(std::unique_ptr<Matcher>&& m, std::unique_ptr<Matcher>&& deep);

  FilterAnalysis precheck(Value* in, Function* start, const FPath&);

  FilterAnalysis decl(CallSite current, const Path& p) const;

  FilterAnalysis def(CallSite current, const Path& p) const;
};

using StandardForwardFilter = BaseFilter<ForwardFilterImpl, DefaultSearch, omp::OmpContext>;

}  // namespace typeart::filter

#endif  // TYPEART_STDFORWARDFILTER_H

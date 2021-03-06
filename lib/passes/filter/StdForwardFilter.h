//
// Created by ahueck on 26.10.20.
//

#ifndef TYPEART_STDFORWARDFILTER_H
#define TYPEART_STDFORWARDFILTER_H

#include "FilterBase.h"
#include "Matcher.h"
#include "filter/IRPath.h"

#include "llvm/IR/CallSite.h"

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

struct StdFilterTrait {
  constexpr static bool Indirect    = false;
  constexpr static bool Intrinsic   = false;
  constexpr static bool Declaration = true;
  constexpr static bool Definition  = true;
  constexpr static bool PreCheck    = true;
};

struct ForwardFilterImpl {
  using Support = StdFilterTrait;
  std::unique_ptr<Matcher> matcher;
  std::unique_ptr<Matcher> deep_matcher;
  FunctionOracleMatcher oracle;  // TODO make set flexible

  explicit ForwardFilterImpl(std::unique_ptr<Matcher>&& m);

  ForwardFilterImpl(std::unique_ptr<Matcher>&& m, std::unique_ptr<Matcher>&& deep);

  FilterAnalysis precheck(Value* in, Function* start, const FPath&);

  FilterAnalysis decl(CallSite current, const Path& p) const;

  FilterAnalysis def(CallSite current, const Path& p) const;
};

using StandardForwardFilter = BaseFilter<ForwardFilterImpl, DefaultSearch, omp::OmpContext>;

}  // namespace typeart::filter

#endif  // TYPEART_STDFORWARDFILTER_H

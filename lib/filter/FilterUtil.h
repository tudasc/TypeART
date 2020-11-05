//
// Created by ahueck on 26.10.20.
//

#ifndef TYPEART_FILTERUTIL_H
#define TYPEART_FILTERUTIL_H

#include "IRPath.h"

#include "llvm/IR/CallSite.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace typeart::filter {

struct FunctionAnalysis {
  using FunctionCounts = struct { int decl, def, intrinsic, indirect; };
  using FunctionCalls  = struct { llvm::SmallVector<CallSite, 8> decl, def, intrinsic, indirect; };

  FunctionCalls calls;

  void clear();

  bool empty() const;

  FunctionCounts analyze(Function* f);
};

raw_ostream& operator<<(raw_ostream& os, const FunctionAnalysis::FunctionCounts& counts);

struct DefUseQueue {
  llvm::SmallPtrSet<Value*, 16> visited_set;
  llvm::SmallVector<Value*, 16> working_set;
  llvm::SmallVector<CallSite, 8> working_set_calls;

  explicit DefUseQueue(Value* init);

  void reset();

  bool empty() const;

  void addToWorkS(Value* v);

  template <typename Range>
  void addToWork(Range&& values) {
    for (auto v : values) {
      addToWorkS(v);
    }
  }

  Value* peek();
};

enum class ArgCorrelation {
  NoMatch,
  Exact,
  ExactMismatch,
  Global,
  GlobalMismatch,
};

inline std::pair<llvm::Argument*, int> find_arg(CallSite c, const Path& p) {
  auto arg = p.getEndPrev();
  if (!arg) {
    return {nullptr, -1};
  }

  Value* in          = arg.getValue();
  const auto arg_pos = llvm::find_if(c.args(), [&in](const Use& arg_use) -> bool { return arg_use.get() == in; });
  if (arg_pos == c.arg_end()) {
    return {nullptr, -1};
  }
  const auto argNum  = std::distance(c.arg_begin(), arg_pos);
  Argument& argument = *(c.getCalledFunction()->arg_begin() + argNum);

  return {&argument, argNum};
}

inline std::vector<llvm::Argument*> args(CallSite c, const Path& p) {
  auto [arg, _] = find_arg(c, p);
  if (arg) {
    return {arg};
  }

  std::vector<llvm::Argument*> args;
  llvm::for_each(c.args(), [&args](llvm::Use& use) {
    Argument* a = llvm::dyn_cast<Argument>(use.get());
    args.emplace_back(a);
  });
  return args;
}

namespace detail {
template <typename TypeID>
ArgCorrelation correlate(CallSite c, const Path& p, TypeID&& isType) {
  auto [arg, _] = find_arg(c, p);

  if (!arg) {
    const auto count_type_ptr = llvm::count_if(c.args(), [&](const auto& arg) {
      const auto type = arg->getType();
      return isType(type);
    });
    if (count_type_ptr > 0) {
      return ArgCorrelation::Global;
    }
    return ArgCorrelation::GlobalMismatch;
  }

  auto type = arg->getType();

  if (isType(type)) {
    return ArgCorrelation::Exact;
  } else {
    return ArgCorrelation::ExactMismatch;
  }
}
}  // namespace detail

inline ArgCorrelation correlate2void(CallSite c, const Path& p) {
  return detail::correlate(
      c, p, [](llvm::Type* type) { return type->isPointerTy() && type->getPointerElementType()->isIntegerTy(8); });
}

}  // namespace typeart::filter

#endif  // TYPEART_FILTERUTIL_H

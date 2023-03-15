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

#ifndef TYPEART_MATCHER_H
#define TYPEART_MATCHER_H

#include "../analysis/MemOpData.h"
#include "../support/Util.h"
#include "compat/CallSite.h"

#include "llvm/ADT/StringSet.h"

namespace typeart::filter {

class Matcher {
 public:
  enum class MatchResult : int { Match, NoMatch, ShouldSkip, ShouldContinue };
  Matcher()               = default;
  Matcher(const Matcher&) = default;
  Matcher(Matcher&&)      = default;
  Matcher& operator=(const Matcher&) = default;
  Matcher& operator=(Matcher&&) = default;

  [[nodiscard]] virtual MatchResult match(llvm::CallSite) const = 0;

  [[nodiscard]] virtual MatchResult match(const llvm::CallBase&) const = 0;

  [[nodiscard]] virtual MatchResult match(const llvm::Function&) const = 0;

  [[nodiscard]] virtual MatchResult match(const llvm::StringRef&) const = 0;

  virtual ~Matcher() = default;
};

namespace detail {
template <Matcher::MatchResult Result>
class StaticMatcher final : public Matcher {
 public:
  [[nodiscard]] MatchResult match(llvm::CallSite) const noexcept override {
    return Result;
  }

  [[nodiscard]] MatchResult match(const llvm::CallBase&) const noexcept override {
    return Result;
  }

  [[nodiscard]] Matcher::MatchResult match(const llvm::Function&) const noexcept override {
    return Result;
  };

  [[nodiscard]] Matcher::MatchResult match(const llvm::StringRef&) const noexcept override {
    return Result;
  };
};
}  // namespace detail

using NoMatcher  = detail::StaticMatcher<Matcher::MatchResult::NoMatch>;
using AnyMatcher = detail::StaticMatcher<Matcher::MatchResult::Match>;

class DefaultStringMatcher final : public Matcher {
  Regex matcher_;

 public:
  explicit DefaultStringMatcher(const std::string& regex) : matcher_(regex, Regex::NoFlags) {
  }
  [[nodiscard]] MatchResult match(llvm::CallSite site) const noexcept override {
    if (const auto* function = site.getCalledFunction()) {
      return DefaultStringMatcher::match(*function);
    }
    return MatchResult::NoMatch;
  }

  [[nodiscard]] MatchResult match(const llvm::CallBase& site) const noexcept override {
    if (const auto* function = site.getCalledFunction()) {
      return DefaultStringMatcher::match(*function);
    }
    return MatchResult::NoMatch;
  }

  [[nodiscard]] MatchResult match(const llvm::Function& function) const noexcept override {
    return DefaultStringMatcher::match(function.getName());
  }

  [[nodiscard]] MatchResult match(const llvm::StringRef& function) const noexcept override {
    const auto f_name  = util::demangle(function);
    const bool matched = matcher_.match(f_name);
    if (!matched) {
      return MatchResult::NoMatch;
    }
    return MatchResult::Match;
  }
};

class FunctionOracleMatcher final : public Matcher {
  const MemOps mem_operations_{};
  llvm::SmallDenseSet<llvm::StringRef> continue_set_{{"sqrt"}, {"cos"}, {"sin"},    {"pow"},  {"fabs"},
                                                     {"abs"},  {"log"}, {"fscanf"}, {"cbrt"}, {"gettimeofday"}};
  llvm::SmallDenseSet<llvm::StringRef> skip_set_{{"printf"}, {"sprintf"},      {"snprintf"}, {"fprintf"},
                                                 {"puts"},   {"__cxa_atexit"}, {"fopen"},    {"fclose"},
                                                 {"scanf"},  {"strtol"},       {"srand"}};

 public:
  [[nodiscard]] MatchResult match(llvm::CallSite site) const noexcept override {
    if (const auto* function = site.getCalledFunction()) {
      return FunctionOracleMatcher::match(*function);
    }
    return MatchResult::NoMatch;
  }

  [[nodiscard]] MatchResult match(const llvm::CallBase& site) const noexcept override {
    if (const auto* function = site.getCalledFunction()) {
      return FunctionOracleMatcher::match(*function);
    }
    return MatchResult::NoMatch;
  }

  [[nodiscard]] MatchResult match(const llvm::Function& function) const noexcept override {
    return FunctionOracleMatcher::match(function.getName());
  }

  [[nodiscard]] MatchResult match(const llvm::StringRef& function) const noexcept override {
    const auto f_name = util::demangle(function);
    const llvm::StringRef f_name_ref{f_name};
    if (continue_set_.count(f_name) > 0) {
      return MatchResult::ShouldContinue;
    }
    if (skip_set_.count(f_name) > 0) {
      return MatchResult::ShouldSkip;
    }
    if (f_name_ref.startswith("__typeart_")) {
      return MatchResult::ShouldSkip;
    }
    if (mem_operations_.kind(f_name)) {
      return MatchResult::ShouldSkip;
    }
    if (f_name_ref.startswith("__ubsan") || f_name_ref.startswith("__asan") || f_name_ref.startswith("__msan")) {
      return MatchResult::ShouldContinue;
    }
    return MatchResult::NoMatch;
  }
};

}  // namespace typeart::filter

#endif  // TYPEART_MATCHER_H

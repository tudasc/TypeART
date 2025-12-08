// TypeART library
//
// Copyright (c) 2017-2025 TypeART Authors
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
  Matcher()                          = default;
  Matcher(const Matcher&)            = default;
  Matcher(Matcher&&)                 = default;
  Matcher& operator=(const Matcher&) = default;
  Matcher& operator=(Matcher&&)      = default;

  virtual MatchResult match(llvm::CallSite const c) const {
    if (c.getCalledFunction() != nullptr)
      return this->matchName(c.getCalledFunction()->getName());

    return MatchResult::NoMatch;
  }

  virtual MatchResult matchName(llvm::StringRef) const = 0;

  virtual ~Matcher() = default;
};

class NoMatcher final : public Matcher {
 public:
  MatchResult matchName(llvm::StringRef) const {
    return MatchResult::NoMatch;
  };
};

class DefaultStringMatcher final : public Matcher {
  llvm::Regex matcher;

 public:
  explicit DefaultStringMatcher(const std::string& regex) : matcher(regex, llvm::Regex::NoFlags) {
  }

  MatchResult matchName(llvm::StringRef const name) const override {
    const auto f_name  = util::demangle(name);
    const bool matched = matcher.match(f_name);

    if (matched) {
      return MatchResult::Match;
    }

    return MatchResult::NoMatch;
  }
};

class FunctionOracleMatcher final : public Matcher {
  const MemOps mem_operations{};
  llvm::SmallDenseSet<llvm::StringRef> continue_set{{"sqrt"}, {"cos"},          {"sin"},    {"pow"},
                                                    {"fabs"}, {"abs"},          {"log"},    {"fscanf"},
                                                    {"cbrt"}, {"gettimeofday"}, {"strcpy"}, {"strlen"}};
  llvm::SmallDenseSet<llvm::StringRef> skip_set{{"printf"}, {"sprintf"},      {"snprintf"}, {"fprintf"},
                                                {"puts"},   {"__cxa_atexit"}, {"fopen"},    {"fclose"},
                                                {"scanf"},  {"strtol"},       {"srand"}};

 public:
  MatchResult matchName(llvm::StringRef const name) const override {
    const auto f_name = util::demangle(name);
    llvm::StringRef f_name_ref{f_name};

    if (continue_set.count(f_name) > 0) {
      return MatchResult::ShouldContinue;
    }
    if (skip_set.count(f_name) > 0) {
      return MatchResult::ShouldSkip;
    }
    if (util::starts_with_any_of(f_name_ref, "__typeart_")) {
      return MatchResult::ShouldSkip;
    }
    if (mem_operations.kind(f_name)) {
      return MatchResult::ShouldSkip;
    }
    if (util::starts_with_any_of(f_name_ref, "__ubsan", "__asan", "__msan", "llvm.", "__tysan", "__dfsan", "__tsan")) {
      return MatchResult::ShouldContinue;
    }

    return MatchResult::NoMatch;
  }
};

}  // namespace typeart::filter

#endif  // TYPEART_MATCHER_H

//
// Created by ahueck on 18.11.20.
//

#ifndef TYPEART_MATCHER_H
#define TYPEART_MATCHER_H

#include "../support/Util.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/IR/CallSite.h"

namespace typeart::filter {

class Matcher {
 public:
  enum class MatchResult : int { Match, NoMatch, ShouldSkip, ShouldContinue };
  Matcher()               = default;
  Matcher(const Matcher&) = default;
  Matcher(Matcher&&)      = default;
  Matcher& operator=(const Matcher&) = default;
  Matcher& operator=(Matcher&&) = default;

  virtual MatchResult match(llvm::CallSite) const = 0;

  virtual ~Matcher() = default;
};

class NoMatcher final : public Matcher {
 public:
  MatchResult match(llvm::CallSite) const {
    return MatchResult::NoMatch;
  };
};

class DefaultStringMatcher final : public Matcher {
  Regex matcher;

 public:
  explicit DefaultStringMatcher(std::string regex) : matcher(std::move(regex), Regex::NoFlags) {
  }

  MatchResult match(llvm::CallSite c) const override {
    const auto f = c.getCalledFunction();
    if (f) {
      const auto f_name  = util::demangle(f->getName());
      const bool matched = matcher.match(f_name);
      if (matched) {
        return MatchResult::Match;
      }
    }
    return MatchResult::NoMatch;
  }
};

class FunctionOracleMatcher final : public Matcher {
  llvm::SmallDenseSet<llvm::StringRef> continue_set{{"sqrt"}, {"cos"}, {"sin"}, {"pow"}, {"fabs"}, {"abs"}, {"log"}};
  llvm::SmallDenseSet<llvm::StringRef> skip_set{{"printf"},       {"sprintf"}, {"snprintf"}, {"fprintf"}, {"puts"},
                                                {"__cxa_atexit"}, {"fopen"},   {"fclose"},   {"scanf"},   {"strcmp"}};

 public:
  MatchResult match(llvm::CallSite c) const override {
    const auto f = c.getCalledFunction();
    if (f) {
      const auto f_name = util::demangle(f->getName());
      if (continue_set.count(f_name) > 0) {
        return MatchResult::ShouldContinue;
      }
      if (skip_set.count(f_name) > 0) {
        return MatchResult::ShouldSkip;
      }
    }
    return MatchResult::NoMatch;
  }
};

}  // namespace typeart::filter

#endif  // TYPEART_MATCHER_H

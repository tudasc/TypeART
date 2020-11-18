//
// Created by ahueck on 18.11.20.
//

#ifndef TYPEART_MATCHER_H
#define TYPEART_MATCHER_H

#include "../support/Util.h"

#include "llvm/IR/CallSite.h"

namespace typeart::filter {

class Matcher {
 public:
  virtual bool match(llvm::CallSite) const = 0;
  virtual ~Matcher()                       = default;
};

class DefaultStringMatcher final : public Matcher {
  Regex matcher;

 public:
  explicit DefaultStringMatcher(std::string regex) : matcher(regex, Regex::NoFlags) {
  }

  bool match(llvm::CallSite c) const override {
    const auto f = c.getCalledFunction();
    if (f) {
      const auto f_name = util::demangle(f->getName());
      return matcher.match(f_name);
    }
    return false;
  }

  DefaultStringMatcher(const DefaultStringMatcher&) = default;
  DefaultStringMatcher(DefaultStringMatcher&&)      = default;
  DefaultStringMatcher& operator=(const DefaultStringMatcher&) = default;
  DefaultStringMatcher& operator=(DefaultStringMatcher&&) = default;
  virtual ~DefaultStringMatcher()                         = default;
};

}  // namespace typeart::filter

#endif  // TYPEART_MATCHER_H

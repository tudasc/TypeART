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
  Matcher()               = default;
  Matcher(const Matcher&) = default;
  Matcher(Matcher&&)      = default;
  Matcher& operator=(const Matcher&) = default;
  Matcher& operator=(Matcher&&) = default;

  virtual bool match(llvm::CallSite) const = 0;

  virtual ~Matcher() = default;
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
};

}  // namespace typeart::filter

#endif  // TYPEART_MATCHER_H

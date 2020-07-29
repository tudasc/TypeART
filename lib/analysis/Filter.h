//
// Created by ahueck on 28.07.20.
//

#ifndef TYPEART_FILTER_H
#define TYPEART_FILTER_H

#include "../support/Util.h"

#include "llvm/IR/Function.h"
namespace llvm {
class Value;
}  // namespace llvm

namespace typeart {
namespace filter {

class FilterBase {
 protected:
  const std::string call_regex;
  bool malloc_mode{false};
  llvm::Function* start_f{nullptr};
  int depth{0};
  std::string reason_trace{""};
  llvm::raw_string_ostream trace;
  bool CallFilterDeep{false};

 public:
  FilterBase(const std::string& glob, bool cfilter)
      : call_regex(util::glob2regex(glob)), CallFilterDeep(cfilter), trace(reason_trace) {
  }

  virtual void setMode(bool search_malloc) {
    malloc_mode = search_malloc;
  }

  llvm::raw_string_ostream& append_trace(std::string s) {
    trace << " | " << s;
    return trace;
  }

  std::string reason() {
    return trace.str();
  }

  void clear_trace() {
    reason_trace.clear();
  }

  virtual void setStartingFunction(llvm::Function* start) {
    start_f = start;
    depth   = 0;
  }

  virtual bool filter(llvm::Value* in) {
    append_trace("NOOP FILTER");
    return false;
  };

  inline std::string getName(const llvm::Function* f) {
    auto name = f->getName();
    // FIXME figure out if we need to demangle, i.e., source is .c or .cpp
    const auto f_name = util::demangle(name);
    if (f_name != "") {
      name = f_name;
    }

    return name;
  }

  virtual ~FilterBase() = default;
};
}  // namespace filter
}  // namespace typeart

#endif  // TYPEART_FILTER_H

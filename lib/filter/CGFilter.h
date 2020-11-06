//
// Created by ahueck on 20.10.20.
//

#ifndef TYPEART_CGFILTER_H
#define TYPEART_CGFILTER_H

#include "../support/Logger.h"
#include "../support/Util.h"
#include "Filter.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"

namespace typeart::filter {

class CGInterface;

namespace deprecated {

using namespace llvm;
class CGFilter final : public Filter {
  const std::string call_regex;
  bool malloc_mode{false};
  llvm::Function* start_f{nullptr};
  int depth{0};
  bool CallFilterDeep{true};
  std::string reason_trace{""};
  llvm::raw_string_ostream trace;
  // Holds pointer to a CG implementation
  std::unique_ptr<CGInterface> callGraph;

 public:
  explicit CGFilter(const std::string& glob, bool CallFilterDeep, std::string file);

  bool filter(Value* in) override;

  void setMode(bool search_malloc) override;

  void setStartingFunction(llvm::Function* start) override;

 private:
  bool shouldContinue(CallSite c, Value* in) const;

  std::string getName(const Function* f);

  llvm::raw_string_ostream& append_trace(std::string s);

  std::string reason();

  void clear_trace();

 public:
  virtual ~CGFilter() = default;
};

}  // namespace deprecated
}  // namespace typeart::filter

#endif  // TYPEART_CGFILTER_H
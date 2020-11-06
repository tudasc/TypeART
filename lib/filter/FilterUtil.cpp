//
// Created by ahueck on 26.10.20.
//

#include "FilterUtil.h"

namespace typeart::filter {

void FunctionAnalysis::clear() {
  calls.indirect.clear();
  calls.def.clear();
  calls.decl.clear();
  calls.intrinsic.clear();
}

bool FunctionAnalysis::empty() const {
  return calls.def.empty() && calls.decl.empty() && calls.indirect.empty() && calls.intrinsic.empty();
}

FunctionAnalysis::FunctionCounts FunctionAnalysis::analyze(Function* f) {
  FunctionCounts count{0, 0, 0, 0};

  for (auto& BB : *f) {
    for (auto& I : BB) {
      CallSite site(&I);
      if (site.isCall()) {
        const auto callee        = site.getCalledFunction();
        const bool indirect_call = callee == nullptr;

        if (indirect_call) {
          ++count.indirect;
          calls.indirect.push_back(site);
          continue;
        }

        const bool is_decl      = callee->isDeclaration();
        const bool is_intrinsic = site.getIntrinsicID() != Intrinsic::not_intrinsic;

        if (is_intrinsic) {
          ++count.intrinsic;
          calls.intrinsic.push_back(site);
          continue;
        }

        if (is_decl) {
          ++count.decl;
          calls.decl.push_back(site);
          continue;
        }

        ++count.def;
        calls.def.push_back(site);
        continue;
      }
    }
  }

  return count;
}

raw_ostream& operator<<(raw_ostream& os, const FunctionAnalysis::FunctionCounts& counts) {
  os << "[ decl:" << counts.decl << ";def:" << counts.def << ";intr:" << counts.intrinsic
     << ";indir:" << counts.indirect << " ]";
  return os;
}
}  // namespace typeart::filter
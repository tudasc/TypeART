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

DefUseQueue::DefUseQueue(Value* init) {
  working_set.emplace_back(init);
}

void DefUseQueue::reset() {
  visited_set.clear();
  working_set.clear();
  working_set_calls.clear();
}

bool DefUseQueue::empty() const {
  return working_set.empty();
}

void DefUseQueue::addToWorkS(Value* v) {
  if (v != nullptr && visited_set.find(v) == visited_set.end()) {
    working_set.push_back(v);
    visited_set.insert(v);
  }
}

Value* DefUseQueue::peek() {
  if (working_set.empty()) {
    return nullptr;
  }
  auto user_iter = working_set.end() - 1;
  working_set.erase(user_iter);
  return *user_iter;
}

}  // namespace typeart::filter
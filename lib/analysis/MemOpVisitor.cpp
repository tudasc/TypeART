/*
 * MemOpVisitor.cpp
 *
 *  Created on: Jan 3, 2018
 *      Author: ahueck
 */

#include "MemOpVisitor.h"
#include "support/Logger.h"
#include "support/TypeUtil.h"

#include "support/Util.h"

#include <algorithm>

namespace typeart {

using namespace llvm;

MemOpVisitor::MemOpVisitor() = default;

void MemOpVisitor::visitCallInst(llvm::CallInst& ci) {
  const auto isInSet = [&](const auto& fMap) -> llvm::Optional<MemOpKind> {
    const auto* f = ci.getCalledFunction();
    if (!f) {
      // TODO handle calls through, e.g., function pointers? - seems infeasible
      LOG_INFO("Encountered indirect call, skipping.");
      return None;
    }
    const auto name = f->getName().str();
    const auto res = fMap.find(name);
    if (res != fMap.end()) {
      return {(*res).second};
    }
    return None;
  };

  if (auto val = isInSet(allocMap)) {
    visitMallocLike(ci, val.getValue());
  } else if (auto val = isInSet(deallocMap)) {
    visitFreeLike(ci, val.getValue());
  }
}

void MemOpVisitor::visitMallocLike(llvm::CallInst& ci, MemOpKind k) {
  //  LOG_DEBUG("Found malloc-like: " << ci.getCalledFunction()->getName());

  SmallPtrSet<BitCastInst*, 4> bcasts;

  for (auto user : ci.users()) {
    // Simple case: Pointer is immediately casted
    if (auto inst = dyn_cast<BitCastInst>(user)) {
      bcasts.insert(inst);
    }
    // Pointer is first stored, then loaded and subsequently casted
    if (auto storeInst = dyn_cast<StoreInst>(user)) {
      auto storeAddr = storeInst->getPointerOperand();
      for (auto storeUser : storeAddr->users()) {  // TODO: Ensure that load occurs ofter store?
        if (auto loadInst = dyn_cast<LoadInst>(storeUser)) {
          for (auto loadUser : loadInst->users()) {
            if (auto bcastInst = dyn_cast<BitCastInst>(loadUser)) {
              bcasts.insert(bcastInst);
            }
          }
        }
      }
    }
  }

  const auto bitcast_iter =
      std::find_if(bcasts.begin(), bcasts.end(), [](auto bcast) { return !util::type::isVoidPtr(bcast->getDestTy()); });

  BitCastInst* primaryBitcast = bitcast_iter != bcasts.end() ? *bitcast_iter : nullptr;

  //  LOG_DEBUG("  >> number of bitcasts found: " << bcasts.size());

  listMalloc.push_back(MallocData{&ci, primaryBitcast, bcasts, k});
}

void MemOpVisitor::visitFreeLike(llvm::CallInst& ci, MemOpKind) {
  //  LOG_DEBUG(ci.getCalledFunction()->getName());

  listFree.insert(&ci);
}

// void MemOpVisitor::visitIntrinsicInst(llvm::IntrinsicInst& ii) {
//
//}

void MemOpVisitor::visitAllocaInst(llvm::AllocaInst& ai) {
  //  LOG_DEBUG("Found alloca " << ai);
  llvm::SmallPtrSet<llvm::CallInst*, 2> lifetimes;

  llvm::SmallPtrSet<Value*, 16> visited_set;
  llvm::SmallVector<Value*, 16> working_set;

  const auto addToWork = [&visited_set, &working_set](auto vals) {
    for (auto v : vals) {
      if (visited_set.find(v) == visited_set.end()) {
        working_set.push_back(v);
        visited_set.insert(v);
      }
    }
  };

  const auto peek = [&working_set]() -> Value* {
    auto user_iter = working_set.end() - 1;
    working_set.erase(user_iter);
    return *user_iter;
  };

  addToWork(ai.users());
  while (!working_set.empty()) {
    auto val = peek();
    if (IntrinsicInst* ii = llvm::dyn_cast<IntrinsicInst>(val)) {
      if (ii->getIntrinsicID() == Intrinsic::lifetime_start
          /* || ii->getIntrinsicID() == Intrinsic::lifetime_end*/) {
        lifetimes.insert(ii);
      }
      continue;
    } else if (llvm::isa<BitCastInst>(val)) {
      addToWork(val->users());  // lifetimes usually get bitcasts passed
    }
  }

  listAlloca.push_back({&ai, lifetimes});
  LOG_ERROR("Alloca: " << util::dump(ai) << " -> lifetime marker: " << util::dump(lifetimes));
}

void MemOpVisitor::clear() {
  listAlloca.clear();
  listMalloc.clear();
  listFree.clear();
}

MemOpVisitor::~MemOpVisitor() = default;

}  // namespace typeart

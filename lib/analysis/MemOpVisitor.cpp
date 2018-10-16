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
namespace finder {
using namespace llvm;

MemOpVisitor::MemOpVisitor() = default;

void MemOpVisitor::visitModuleGlobals(Module& m) {
  for (auto& g : m.globals()) {
    listGlobals.push_back(&g);
  }
}

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

  BitCastInst* primaryBitcast{nullptr};
  std::for_each(bcasts.begin(), bcasts.end(), [&](auto bcast) {
    if (!util::type::isVoidPtr(bcast->getDestTy())) {
      primaryBitcast = bcast;
    }
  });

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
  Value* arraySizeOperand = ai.getArraySize();
  int arraySize = -1;
  if (auto arraySizeConst = llvm::dyn_cast<ConstantInt>(arraySizeOperand)) {
    arraySize = arraySizeConst->getZExtValue();
  }

  listAlloca.push_back({&ai, arraySize});
  //  LOG_DEBUG("Alloca: " << util::dump(ai) << " -> lifetime marker: " << util::dump(lifetimes));
}  // namespace typeart

void MemOpVisitor::clear() {
  listAlloca.clear();
  listMalloc.clear();
  listFree.clear();
}

MemOpVisitor::~MemOpVisitor() = default;

}  // namespace finder
}  // namespace typeart

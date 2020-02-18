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

#include "llvm/IR/InstrTypes.h"

namespace typeart {
namespace finder {
using namespace llvm;

template<typename Map>
auto isInSetCheck(const Map &fMap, llvm::Instruction &i) -> Optional<typename Map::mapped_type> {
    llvm::Function *f = nullptr;
    if (llvm::isa<llvm::CallInst>(i)) {
        f = llvm::cast<CallInst>(i).getCalledFunction();
    } else if (llvm::isa<llvm::InvokeInst>(i)) {
        f = llvm::cast<InvokeInst>(i).getCalledFunction();
    }
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

}

MemOpVisitor::MemOpVisitor() = default;

void MemOpVisitor::visitModuleGlobals(Module& m) {
  for (auto& g : m.globals()) {
    listGlobals.push_back(&g);
  }
}

void MemOpVisitor::visitCallInst(llvm::CallInst& ci) {
  if (auto val = isInSetCheck(allocMap,ci)) {
    visitMallocLike(ci, val.getValue());
  } else if (auto val = isInSetCheck(deallocMap,ci)) {
    visitFreeLike(ci, val.getValue());
  } else if (auto val = isInSetCheck(assertMap,ci)) {
    visitTypeAssert(ci, val.getValue());
  }
}

void MemOpVisitor::visitInvokeInst(llvm::InvokeInst& ii) {
  if (auto val = isInSetCheck(allocMap, ii)) {
    LOG_WARNING("visiting malloc-like invoke instruction");
  } else if (auto val = isInSetCheck(deallocMap, ii)) {
    LOG_WARNING("visiting free-like invoke instruction");
  } else if (auto val = isInSetCheck(assertMap, ii)) {
    visitTypeAssert(ii, val.getValue());
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

void MemOpVisitor::visitTypeAssert(CallInst& ci, AssertKind k) {
    LOG_INFO("visiting" << ci );
  listAssert.push_back({&ci, k});
}

void MemOpVisitor::visitTypeAssert(InvokeInst& ii, AssertKind k) {
    LOG_INFO("visiting" << ii );
    listAssert.push_back({&ii, k});
}

void MemOpVisitor::clear() {
  listAlloca.clear();
  listMalloc.clear();
  listFree.clear();
  listAssert.clear();
}

MemOpVisitor::~MemOpVisitor() = default;

}  // namespace finder
}  // namespace typeart

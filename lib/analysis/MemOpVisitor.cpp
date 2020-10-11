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
    globals.emplace_back(GlobalData{&g});
  }
}

void MemOpVisitor::visitCallBase(llvm::CallBase& cb) {
  const auto isInSet = [&](const auto& fMap) -> llvm::Optional<MemOpKind> {
    const auto* f = cb.getCalledFunction();
    if (!f) {
      // TODO handle calls through, e.g., function pointers? - seems infeasible
      LOG_INFO("Encountered indirect call, skipping.");
      return None;
    }
    const auto name = f->getName().str();
    const auto res  = fMap.find(name);
    if (res != fMap.end()) {
      return {(*res).second};
    }
    return None;
  };

  if (auto val = isInSet(alloc_map)) {
    visitMallocLike(cb, val.getValue());
  } else if (auto val = isInSet(dealloc_map)) {
    visitFreeLike(cb, val.getValue());
  }
}

void MemOpVisitor::visitMallocLike(llvm::CallBase& ci, MemOpKind k) {
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
              LOG_MSG(*bcastInst)
              bcasts.insert(bcastInst);
            }
          }
        }
      }
    }
    // FIXME this is a try to fix issue #13 ; may require more sophisticated dataflow tracking
    if (auto gep = dyn_cast<GetElementPtrInst>(user)) {
      // if (gep->getPointerOperand() == ci) {
      for (auto gep_user : gep->users()) {
        if (auto bcastInst = dyn_cast<BitCastInst>(gep_user)) {
          bcasts.insert(bcastInst);
        }
      }
      //}
    }
  }

  BitCastInst* primary_cast{nullptr};
  if (!bcasts.empty()) {
    primary_cast = *bcasts.begin();
  }

  using namespace util::type;
  // const auto is_i64 = [](auto* type) { return type->isPointerTy() && type->getPointerElementType()->isIntegerTy(64);
  // };

  const bool has_specific = llvm::count_if(bcasts, [&](auto bcast) {
                              auto dest = bcast->getDestTy();
                              return !isVoidPtr(dest) && !isi64Ptr(dest);
                            }) > 0;

  for (auto bcast : bcasts) {
    auto dest = bcast->getDestTy();
    if (!isVoidPtr(dest)) {
      auto cast = bcast;
      if (isi64Ptr(dest) && has_specific) {  // LLVM likes to treat mallocs with i64 ptr type, skip if we have sth else
        continue;
      }
      primary_cast = cast;
    }
  }

  if (primary_cast == nullptr) {
    LOG_DEBUG("Primay bitcast null: " << ci)
  }
  /*
  std::for_each(bcasts.begin(), bcasts.end(), [&](auto bcast) {
    using namespace util::type;
    auto dest = bcast->getDestTy();
    if (!isVoidPtr(dest)) {
      primary_cast = bcast;
    }
  });
  */
  //  LOG_DEBUG("  >> number of bitcasts found: " << bcasts.size());

  mallocs.push_back(MallocData{&ci, primary_cast, bcasts, k, isa<InvokeInst>(ci)});
}

void MemOpVisitor::visitFreeLike(llvm::CallBase& ci, MemOpKind) {
  //  LOG_DEBUG(ci.getCalledFunction()->getName());

  frees.emplace_back(FreeData{&ci, isa<InvokeInst>(ci)});
}

// void MemOpVisitor::visitIntrinsicInst(llvm::IntrinsicInst& ii) {
//
//}

void MemOpVisitor::visitAllocaInst(llvm::AllocaInst& ai) {
  //  LOG_DEBUG("Found alloca " << ai);
  Value* arraySizeOperand = ai.getArraySize();
  size_t arraySize{0};
  bool is_vla{false};
  if (auto arraySizeConst = llvm::dyn_cast<ConstantInt>(arraySizeOperand)) {
    arraySize = arraySizeConst->getZExtValue();
  } else {
    is_vla = true;
  }

  allocas.push_back({&ai, arraySize, is_vla});
  //  LOG_DEBUG("Alloca: " << util::dump(ai) << " -> lifetime marker: " << util::dump(lifetimes));
}  // namespace typeart

void MemOpVisitor::clear() {
  allocas.clear();
  mallocs.clear();
  frees.clear();
}

MemOpVisitor::~MemOpVisitor() = default;

}  // namespace finder
}  // namespace typeart

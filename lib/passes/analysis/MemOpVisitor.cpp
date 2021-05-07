/*
 * MemOpVisitor.cpp
 *
 *  Created on: Jan 3, 2018
 *      Author: ahueck
 */

#include "MemOpVisitor.h"

#include "analysis/MemOpData.h"
#include "support/Logger.h"
#include "support/TypeUtil.h"

#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>

namespace typeart::finder {

using namespace llvm;

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
      // LOG_INFO("Encountered indirect call, skipping.");
      return None;
    }
    const auto name = f->getName().str();

    const auto res = fMap.find(name);
    if (res != fMap.end()) {
      return {(*res).second};
    }
    return None;
  };

  if (auto val = isInSet(mem_operations.allocs())) {
    visitMallocLike(cb, val.getValue());
  } else if (auto val = isInSet(mem_operations.deallocs())) {
    visitFreeLike(cb, val.getValue());
  }
}

void MemOpVisitor::visitMallocLike(llvm::CallBase& ci, MemOpKind k) {
  //  LOG_DEBUG("Found malloc-like: " << ci.getCalledFunction()->getName());

  SmallPtrSet<GetElementPtrInst*, 2> geps;
  SmallPtrSet<BitCastInst*, 4> bcasts;

  BitCastInst* primary_cast{nullptr};
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
              // LOG_MSG(*bcastInst)
              bcasts.insert(bcastInst);
            }
          }
        }
      }
    }
    // GEP indicates that an array cookie is added to the allocation. (Fixes #13)
    if (auto gep = dyn_cast<GetElementPtrInst>(user)) {
      geps.insert(gep);
    }
  }

  if (!bcasts.empty()) {
    primary_cast = *bcasts.begin();
  }

  // Handle array cookies.
  using namespace util::type;
  if (geps.size() == 1) {
    // The memory allocation has an unpadded array cookie.
    auto gep = *geps.begin();
    for (auto gep_user : gep->users()) {
      if (auto bcast_inst = dyn_cast<BitCastInst>(gep_user)) {
        bcasts.insert(bcast_inst);
        primary_cast = bcast_inst;
      }
    }
  } else if (geps.size() == 2) {
    // The memory allocation has a padded array cookie.
    auto gep_it     = geps.begin();
    auto cookie_gep = *gep_it++;
    auto gep        = *gep_it;
    for (auto gep_user : gep->users()) {
      if (auto bcast_inst = dyn_cast<BitCastInst>(gep_user)) {
        if (!isi64Ptr(bcast_inst->getDestTy())) {
          bcasts.insert(bcast_inst);
          primary_cast = bcast_inst;
        }
      }
    }
  } else if (geps.size() > 2) {
    // Found a case where the address of an allocation is used more than two
    // times as an argument to a GEP instruction. This is unexpected as at most
    // two GEPs, for calculating the offsets of an array cookie itself and the
    // array pointer, are expected.
    LOG_ERROR("Expected at most two GEP instructions!");
  }

  if (primary_cast == nullptr) {
    LOG_DEBUG("Primay bitcast null: " << ci)
  }

  mallocs.push_back(MallocData{&ci, primary_cast, bcasts, k, isa<InvokeInst>(ci)});
}

void MemOpVisitor::visitFreeLike(llvm::CallBase& ci, MemOpKind k) {
  //  LOG_DEBUG(ci.getCalledFunction()->getName());
  MemOpKind kind = k;

  // FIME is that superfluous?
  if (auto f = ci.getCalledFunction()) {
    auto dkind = mem_operations.deallocKind(f->getName());
    if (dkind) {
      kind = dkind.getValue();
    }
  }

  frees.emplace_back(FreeData{&ci, kind, isa<InvokeInst>(ci)});
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

}  // namespace typeart::finder

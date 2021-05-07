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

template <class T>
auto expect_single_user(llvm::Value* value) -> T* {
  auto users    = value->users();
  auto users_it = users.begin();
  auto first    = *users_it++;
  assert(users_it == users.end());
  assert(isa<T>(first));
  return dyn_cast<T>(first);
};

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
  auto const array_cookie_from = [](llvm::GetElementPtrInst* array_gep, llvm::BitCastInst* inst) -> ArrayCookieData {
    assert(isi64Ptr(inst->getDestTy()));
    auto cookie_store = expect_single_user<StoreInst>(inst);
    assert(array_gep->getNumIndices() == 1);
    return ArrayCookieData{cookie_store, array_gep->getOperand(1)};
  };

  auto array_cookie = llvm::Optional<ArrayCookieData>{};
  if (geps.size() == 1) {
    // We expect only the bitcast to size_t for the array cookie store.
    assert(bcasts.size() == 1);

    // The memory allocation has an unpadded array cookie.
    auto array_gep   = *geps.begin();
    auto array_bcast = expect_single_user<BitCastInst>(array_gep);
    bcasts.insert(array_bcast);
    primary_cast = array_bcast;

    // In case of an unpadded array cookie we expect a single bitcast to size_t.
    for (auto const& bcast : bcasts) {
      bcast->dump();
    }
    array_cookie = {array_cookie_from(array_gep, *bcasts.begin())};
  } else if (geps.size() == 2) {
    // We expect bitcasts only after the GEP instructions in this case.
    assert(bcasts.size() == 0);

    // The memory allocation has a padded array cookie.
    auto gep_it      = geps.begin();
    auto cookie_gep  = *gep_it++;
    auto array_gep   = *gep_it;
    auto array_bcast = expect_single_user<BitCastInst>(array_gep);
    if (!isi64Ptr(array_bcast->getDestTy())) {
      bcasts.insert(array_bcast);
      primary_cast = array_bcast;
    }
    array_cookie = {array_cookie_from(cookie_gep, expect_single_user<BitCastInst>(cookie_gep))};
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

  mallocs.push_back(MallocData{&ci, array_cookie, primary_cast, bcasts, k, isa<InvokeInst>(ci)});
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

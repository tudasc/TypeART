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
#include "llvm/IR/CallSite.h"
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

MemOpVisitor::MemOpVisitor() : MemOpVisitor(true, true) {
}

MemOpVisitor::MemOpVisitor(bool collectAllocas, bool collectHeap)
    : collectAllocas(collectAllocas), collectHeap(collectHeap) {
}

void MemOpVisitor::visitModuleGlobals(Module& m) {
  for (auto& g : m.globals()) {
    globals.emplace_back(GlobalData{&g});
  }
}

void MemOpVisitor::visitCallBase(llvm::CallBase& cb) {
  if (!collectHeap) {
    return;
  }
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
T* get_single_user_as(llvm::Value* value) {
  auto users           = value->users();
  const auto num_users = value->getNumUses();

  assert(num_users > 0);

  // Check for calls: In case of ASAN, the array cookie is passed to its API.
  // TODO: this may need to be extended to include other such calls
  const auto num_asan_calls = llvm::count_if(users, [](User* user) {
    CallSite csite(user);
    if (!(csite.isCall() || csite.isInvoke()) || csite.getCalledFunction() == nullptr) {
      return false;
    }
    const auto name = csite.getCalledFunction()->getName();
    return name.startswith("__asan");
  });

  assert(num_asan_calls == (num_users - 1));

  // This should return the type T we need:
  auto user = llvm::find_if(users, [](auto user) { return isa<T>(*user); });
  if (user == std::end(users)) {
    LOG_FATAL("Could not find non-asan-call user of " << *value);
    return nullptr;
  }
  return dyn_cast<T>(*user);
}

using MallocGeps   = SmallPtrSet<GetElementPtrInst*, 2>;
using MallocBcasts = SmallPtrSet<BitCastInst*, 4>;

std::pair<MallocGeps, MallocBcasts> collectRelevantMallocUsers(llvm::CallBase& ci) {
  auto geps   = MallocGeps{};
  auto bcasts = MallocBcasts{};
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
  return {geps, bcasts};
}

llvm::Optional<ArrayCookieData> handleUnpaddedArrayCookie(const MallocGeps& geps, MallocBcasts& bcasts,
                                                          BitCastInst*& primary_cast) {
  using namespace util::type;
  // We expect only the bitcast to size_t for the array cookie store.
  assert(bcasts.size() == 1);
  auto cookie_bcast = *bcasts.begin();
  assert(isi64Ptr(cookie_bcast->getDestTy()));
  auto cookie_store = get_single_user_as<StoreInst>(cookie_bcast);

  auto array_gep = *geps.begin();
  assert(array_gep->getNumIndices() == 1);
  auto array_bcast = get_single_user_as<BitCastInst>(array_gep);
  bcasts.insert(array_bcast);
  primary_cast = array_bcast;

  return {ArrayCookieData{cookie_store, array_gep}};
}

llvm::Optional<ArrayCookieData> handlePaddedArrayCookie(const MallocGeps& geps, MallocBcasts& bcasts,
                                                        BitCastInst*& primary_cast) {
  using namespace util::type;
  // We expect bitcasts only after the GEP instructions in this case.
  assert(bcasts.empty());

  auto gep_it     = geps.begin();
  auto array_gep  = *gep_it++;
  auto cookie_gep = *gep_it++;

  auto cookie_bcast = get_single_user_as<BitCastInst>(cookie_gep);
  assert(isi64Ptr(cookie_bcast->getDestTy()));
  auto cookie_store = get_single_user_as<StoreInst>(cookie_bcast);

  assert(array_gep->getNumIndices() == 1);
  auto array_bcast = get_single_user_as<BitCastInst>(array_gep);
  bcasts.insert(array_bcast);
  primary_cast = array_bcast;

  return {ArrayCookieData{cookie_store, array_gep}};
}

llvm::Optional<ArrayCookieData> handleGepInstrs(const MallocGeps& geps, MallocBcasts& bcasts,
                                                BitCastInst*& primary_cast) {
  if (geps.size() == 1) {
    return handleUnpaddedArrayCookie(geps, bcasts, primary_cast);
  } else if (geps.size() == 2) {
    return handlePaddedArrayCookie(geps, bcasts, primary_cast);
  } else if (geps.size() > 2) {
    // Found a case where the address of an allocation is used more than two
    // times as an argument to a GEP instruction. This is unexpected as at most
    // two GEPs, for calculating the offsets of an array cookie itself and the
    // array pointer, are expected.
    LOG_FATAL("Expected at most two GEP instructions!");
  }
  return llvm::None;
}

void MemOpVisitor::visitMallocLike(llvm::CallBase& ci, MemOpKind k) {
  auto [geps, bcasts] = collectRelevantMallocUsers(ci);
  auto primary_cast   = bcasts.empty() ? nullptr : *bcasts.begin();
  auto array_cookie   = handleGepInstrs(geps, bcasts, primary_cast);
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

  auto gep              = dyn_cast<GetElementPtrInst>(ci.getArgOperand(0));
  auto array_cookie_gep = gep != nullptr ? llvm::Optional<llvm::GetElementPtrInst*>{gep} : llvm::None;
  frees.emplace_back(FreeData{&ci, array_cookie_gep, kind, isa<InvokeInst>(ci)});
}

// void MemOpVisitor::visitIntrinsicInst(llvm::IntrinsicInst& ii) {
//
//}

void MemOpVisitor::visitAllocaInst(llvm::AllocaInst& ai) {
  if (!collectAllocas) {
    return;
  }
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

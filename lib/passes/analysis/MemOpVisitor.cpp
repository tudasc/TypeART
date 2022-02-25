// TypeART library
//
// Copyright (c) 2017-2022 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#include "MemOpVisitor.h"

#include "analysis/MemOpData.h"
#include "compat/CallSite.h"
#include "support/Error.h"
#include "support/Logger.h"
#include "support/TypeUtil.h"

#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#if LLVM_VERSION_MAJOR >= 12
#include "llvm/Analysis/ValueTracking.h"  // llvm::findAllocaForValue
#else
#include "llvm/Transforms/Utils/Local.h"  // llvm::findAllocaForValue
#endif
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>

namespace typeart::analysis {

using namespace llvm;

MemOpVisitor::MemOpVisitor() : MemOpVisitor(true, true) {
}

MemOpVisitor::MemOpVisitor(bool collect_allocas, bool collect_heap)
    : collect_allocas(collect_allocas), collect_heap(collect_heap) {
}

void MemOpVisitor::collect(llvm::Function& function) {
  visit(function);

  for (auto& [lifetime, alloc] : lifetime_starts) {
    auto* data =
        llvm::find_if(allocas, [alloc = std::ref(alloc)](const AllocaData& data) { return data.alloca == alloc; });
    if (data != std::end(allocas)) {
      data->lifetime_start.insert(lifetime);
    }
  }

  for (const auto& alloc : allocas) {
    if (alloc.lifetime_start.size() > 1) {
      LOG_DEBUG("Lifetime: " << alloc.lifetime_start.size());
      LOG_DEBUG(*alloc.alloca);
      for (auto* lifetime : alloc.lifetime_start) {
        LOG_DEBUG(*lifetime);
      }
    }
  }
}

void MemOpVisitor::collectGlobals(Module& module) {
  for (auto& g : module.globals()) {
    globals.emplace_back(GlobalData{&g});
  }
}

void MemOpVisitor::visitCallBase(llvm::CallBase& cb) {
  if (!collect_heap) {
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

  if (auto alloc_val = isInSet(mem_operations.allocs())) {
    visitMallocLike(cb, alloc_val.getValue());
  } else if (auto dealloc_val = isInSet(mem_operations.deallocs())) {
    visitFreeLike(cb, dealloc_val.getValue());
  }
}

template <class T>
llvm::Expected<T*> getSingleUserAs(llvm::Value* value) {
  auto users           = value->users();
  const auto num_users = value->getNumUses();
  RETURN_ERROR_IF(num_users == 0, "Expected a single user on value \"{}\" that has no users!", *value);

  // Check for calls: In case of ASAN, the array cookie is passed to its API.
  // TODO: this may need to be extended to include other such calls
  const auto num_asan_calls = llvm::count_if(users, [](llvm::User* user) {
    CallSite csite(user);
    if (!(csite.isCall() || csite.isInvoke()) || csite.getCalledFunction() == nullptr) {
      return false;
    }
    const auto name = csite.getCalledFunction()->getName();
    return name.startswith("__asan");
  });
  RETURN_ERROR_IF(num_asan_calls != (num_users - 1),
                  "Expected a single user on value \"{}\" but found multiple potential candidates!", *value);

  // This should return the type T we need:
  auto user = llvm::find_if(users, [](auto user) { return isa<T>(*user); });
  RETURN_ERROR_IF(user == std::end(users),
                  "Expected a single user on value \"{}\" but didn't find a user of the desired type!", *value);

  return {dyn_cast<T>(*user)};
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

llvm::Expected<ArrayCookieData> handleUnpaddedArrayCookie(const MallocGeps& geps, MallocBcasts& bcasts,
                                                          BitCastInst*& primary_cast) {
  using namespace util::type;
  // We expect only the bitcast to size_t for the array cookie store.
  RETURN_ERROR_IF(bcasts.size() != 1, "Couldn't identify bitcast instruction of an unpadded array cookie!");

  auto cookie_bcast = *bcasts.begin();
  RETURN_ERROR_IF(!isi64Ptr(cookie_bcast->getDestTy()), "Found non-i64Ptr bitcast instruction for an array cookie!");

  auto cookie_store = getSingleUserAs<StoreInst>(cookie_bcast);
  RETURN_ON_ERROR(cookie_store);

  auto array_gep = *geps.begin();
  RETURN_ERROR_IF(array_gep->getNumIndices() != 1, "Found multidimensional array cookie gep!");

  auto array_bcast = getSingleUserAs<BitCastInst>(array_gep);
  RETURN_ON_ERROR(array_bcast);

  bcasts.insert(*array_bcast);
  primary_cast = *array_bcast;

  return {ArrayCookieData{*cookie_store, array_gep}};
}

llvm::Expected<ArrayCookieData> handlePaddedArrayCookie(const MallocGeps& geps, MallocBcasts& bcasts,
                                                        BitCastInst*& primary_cast) {
  using namespace util::type;
  // We expect bitcasts only after the GEP instructions in this case.
  RETURN_ERROR_IF(!bcasts.empty(), "Found unrelated bitcast instructions on a padded array cookie!");

  auto gep_it     = geps.begin();
  auto array_gep  = *gep_it++;
  auto cookie_gep = *gep_it++;

  auto cookie_bcast = getSingleUserAs<BitCastInst>(cookie_gep);
  RETURN_ON_ERROR(cookie_bcast);
  RETURN_ERROR_IF(!isi64Ptr((*cookie_bcast)->getDestTy()), "Found non-i64Ptr bitcast instruction for an array cookie!");

  auto cookie_store = getSingleUserAs<StoreInst>(*cookie_bcast);
  RETURN_ON_ERROR(cookie_store);
  RETURN_ERROR_IF(array_gep->getNumIndices() != 1, "Found multidimensional array cookie gep!");

  auto array_bcast = getSingleUserAs<BitCastInst>(array_gep);
  RETURN_ON_ERROR(array_bcast);

  bcasts.insert(*array_bcast);
  primary_cast = *array_bcast;

  return {ArrayCookieData{*cookie_store, array_gep}};
}

llvm::Optional<ArrayCookieData> handleArrayCookie(const MallocGeps& geps, MallocBcasts& bcasts,
                                                  BitCastInst*& primary_cast) {
  auto exit_on_error = llvm::ExitOnError{"Array Cookie Detection failed!"};
  if (geps.size() == 1) {
    return exit_on_error(handleUnpaddedArrayCookie(geps, bcasts, primary_cast));
  } else if (geps.size() == 2) {
    return exit_on_error(handlePaddedArrayCookie(geps, bcasts, primary_cast));
  } else if (geps.size() > 2) {
    // Found a case where the address of an allocation is used more than two
    // times as an argument to a GEP instruction. This is unexpected as at most
    // two GEPs, for calculating the offsets of an array cookie itself and the
    // array pointer, are expected.
    auto err = "Expected at most two GEP instructions!";
    LOG_FATAL(err);
    exit_on_error({error::make_string_error(err)});
  }
  return llvm::None;
}

void MemOpVisitor::visitMallocLike(llvm::CallBase& ci, MemOpKind k) {
  auto [geps, bcasts] = collectRelevantMallocUsers(ci);
  auto primary_cast   = bcasts.empty() ? nullptr : *bcasts.begin();
  auto array_cookie   = handleArrayCookie(geps, bcasts, primary_cast);
  if (primary_cast == nullptr) {
    LOG_DEBUG("Primary bitcast null: " << ci)
  }
  mallocs.push_back(MallocData{&ci, array_cookie, primary_cast, bcasts, k, isa<InvokeInst>(ci)});
}

void MemOpVisitor::visitFreeLike(llvm::CallBase& ci, MemOpKind k) {
  //  LOG_DEBUG(ci.getCalledFunction()->getName());
  MemOpKind kind = k;

  // FIXME is that superfluous?
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
  if (!collect_allocas) {
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
}

void MemOpVisitor::visitIntrinsicInst(llvm::IntrinsicInst& inst) {
  if (inst.getIntrinsicID() == Intrinsic::lifetime_start) {
#if LLVM_VERSION_MAJOR >= 12
    auto alloca = llvm::findAllocaForValue(inst.getOperand(1));
#else
    DenseMap<Value*, AllocaInst*> alloca_for_value;
    auto* alloca = llvm::findAllocaForValue(inst.getOperand(1), alloca_for_value);
#endif
    if (alloca != nullptr) {
      lifetime_starts.emplace_back(&inst, alloca);
    }
  }
}

void MemOpVisitor::clear() {
  allocas.clear();
  mallocs.clear();
  frees.clear();
}

}  // namespace typeart::analysis

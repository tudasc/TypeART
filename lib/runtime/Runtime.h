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

#ifndef TYPEART_RUNTIME_H
#define TYPEART_RUNTIME_H

#include "AccessCounter.h"
#include "AllocationTracking.h"
#include "TypeDB.h"
#include "TypeResolution.h"

#include <cstddef>
#include <string>

namespace typeart {
struct PointerInfo;

namespace debug {

std::string toString(const void* memAddr, int typeId, size_t count, size_t typeSize, const void* calledFrom);

std::string toString(const void* memAddr, int typeId, size_t count, const void* calledFrom);

std::string toString(const void* addr, const PointerInfo& info);

}  // namespace debug

struct RuntimeSystem {
 private:
  // rtScope must be set to true before all other members are initialized.
  // This is achieved by adding this struct as the first member.
  struct RTScopeInitializer {
    RTScopeInitializer() : rtScopeWasSet(rtScope) {
      rtScope = true;
    }

    void reset() {
      // Reset rtScope to old value.
      rtScope = rtScopeWasSet;
    }

   private:
    bool rtScopeWasSet;
  };

  RTScopeInitializer rtScopeInit;
  TypeDB typeDB{};

 public:
  Recorder recorder{};
  TypeResolution typeResolution;
  AllocationTracker allocTracker;

  static thread_local bool rtScope;

  static RuntimeSystem& get() {
    // As opposed to a global variable, a singleton + instantiation during
    // the first callback/query avoids some problems when
    // preloading (especially with MUST).
    static RuntimeSystem instance;
    return instance;
  }

 private:
  RuntimeSystem();
  ~RuntimeSystem();
};

struct RTGuard final {
  RTGuard() : alreadyInRT(typeart::RuntimeSystem::rtScope) {
    typeart::RuntimeSystem::rtScope = true;
  }

  ~RTGuard() {
    if (!alreadyInRT) {
      typeart::RuntimeSystem::rtScope = false;
    }
  }

  bool shouldTrack() const {
    return !alreadyInRT;
  }

 private:
  const bool alreadyInRT;
};

}  // namespace typeart

#endif  // TYPEART_RUNTIME_H

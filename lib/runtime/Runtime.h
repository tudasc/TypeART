// TypeART library
//
// Copyright (c) 2017-2026 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#ifndef LIB_RUNTIME_RUNTIME
#define LIB_RUNTIME_RUNTIME

#include "AccessCounter.h"
#include "AllocationTracking.h"
#include "GlobalTypeDefCallbacks.h"
#include "TypeDB.h"
#include "TypeResolution.h"

#include <cstddef>
#include <string>

namespace typeart {
struct PointerInfo;

namespace debug {

std::string toString(const void* memAddr, int typeId, size_t count, size_t typeSize, const void* calledFrom,
                     bool heap = false);

std::string toString(const void* memAddr, int typeId, size_t count, const void* calledFrom, bool heap = false);

std::string toString(const void* addr, const PointerInfo& info, bool heap = false);

}  // namespace debug

struct RuntimeSystem {
 private:
  // rtScope must be set to true before all other members are initialized.
  // This is achieved by adding this struct as the first member.
  struct RTScopeInitializer {
    RTScopeInitializer() : rtScopeWasSet(rtScope) {
      rtScope = true;
    }

    void reset() const {
      // Reset rtScope to old value.
      rtScope = rtScopeWasSet;
    }

   private:
    bool rtScopeWasSet;
  };

  RTScopeInitializer rtScopeInit;
  TypeDB typeDB_{};
  TypeResolution typeResolution_;
  AllocationTracker allocTracker_;
  GlobalTypeTranslator type_translator_;

 public:
  Recorder recorder{};
  static thread_local bool rtScope;

  const TypeDB& database() const {
    return typeDB_;
  }

  TypeResolution& get_type_resolution() {
    return typeResolution_;
  }

  AllocationTracker& allocation_tracker() {
    return allocTracker_;
  }

  GlobalTypeTranslator& type_translator() {
    return type_translator_;
  }

  const GlobalTypeTranslator& type_translator() const {
    return type_translator_;
  }

  const TypeResolution& type_resolution() const {
    return typeResolution_;
  }

  const AllocationTracker& allocation_tracker() const {
    return allocTracker_;
  }

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

#endif /* LIB_RUNTIME_RUNTIME */

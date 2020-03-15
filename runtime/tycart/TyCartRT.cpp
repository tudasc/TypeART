#include "Logger.h"
#include "Runtime.h"
#include "tycart.h"

#include "RuntimeInterface.h"

#include <sstream>

#ifdef WITH_FTI
#include "fti.h"
#endif

#ifdef WITH_VELOC
#include "veloc.h"
#endif

#include <iostream>
#include <map>

namespace tycart {

enum class AssertKind { STRICT, RELAXED };

static AssertKind assert_kind = AssertKind::STRICT;

struct CPData {
  void* addr;
  int typeId;
  size_t count;
};
static std::map<int, CPData> CPs;

inline void insert(int id, CPData cpd) {
  LOG_DEBUG("Inserting id: " << id);
  CPs.insert({id, cpd});
}

inline int remove(int id) {
  LOG_DEBUG("Remove requested for id: " << id);
  auto it = CPs.find(id);
  if (it != CPs.end()) {
    LOG_DEBUG("Removinging id: " << id);
    CPs.erase(it);
    return 0;
  }
  return 1;
}

namespace impl {
inline void _do_protect(int id, void* addr, size_t count, size_t baseSize, int typeId) {
#ifdef WITH_VELOC
  (void)typeId;  // silence compiler warning
  VELOC_Mem_protect(id, addr, count, baseSize);
#endif
#ifdef WITH_FTI
  (void)baseSize;  // silence compiler warning
  FTI_Protect(id, addr, count, fti::getFTIType(typeId));
#endif
}

#ifdef WITH_FTI
namespace fti {
static std::map<int, FTIT_type> FTITs;
static bool initialized{false};

inline void registerFTIType(int typeId) {
  if (!initialized) {
    registerFTIBuiltInTypes();
    initialized = true;
  }
  FTIT_type dType;
  FTI_InitType(&dType, TypeArtRT::get().getTypeSize(typeId));
  // XXX Assumes FTIT_type can be safely copied around.
  FTITs.insert({typeId, dType});
}

inline FTIT_type getFTIType(int typeId) {
  if (!initialized) {
    registerFTIBuiltInTypes();
    initialized = true;
  }
  auto it = FTITs.find(typeId);
  if (it == FTITs.end()) {
    LOG_FATAL("FTI Type not in map for typeId: " << typeId);
    std::exit(EXIT_FAILURE);
  }
  return it->second;
}

void registerFTIBuiltInTypes() {
  // XXX There are some FTI types missing, e.g., unsigned types.
#define TY_REG(id, T) FTITs.insert({id, T});
  TY_REG(0, FTI_CHAR)
  TY_REG(1, FTI_SHRT)
  TY_REG(2, FTI_INTG)
  TY_REG(3, FTI_LONG)
  TY_REG(5, FTI_SFLT)
  TY_REG(6, FTI_DBLE)
  TY_REG(7, FTI_LDBE)
#undef TY_REG
}
}  // namespace fti
#endif  // WITH_FTI
}  // namespace impl

inline void TYdo_assert(void* addr, int typeId, size_t count, AssertKind assertk = AssertKind::STRICT) {
  const auto fail = [&](std::string msg) -> void {
    LOG_FATAL("Assert failed: " << msg);
    exit(EXIT_FAILURE);
  };
  const auto type_mismatch_fail = [&](auto actualTypeId) {
    const char* expectedName = typeart_get_type_name(typeId);
    const char* actualName = typeart_get_type_name(actualTypeId);
    std::stringstream ss;
    ss << "Expected type " << expectedName << "(id=" << typeId << ") but got " << actualName << "(id=" << actualTypeId
       << ")";
    fail(ss.str());
  };
  const auto count_mismatch_fail = [&](auto actualCount) {
    std::stringstream ss;
    ss << "Expected number of elements is " << count << " but actual number is " << actualCount;
    fail(ss.str());
  };

  const auto ta_status = [&fail](auto status) {
    switch (status) {
      case TA_OK:
        break;
      case TA_INVALID_ID:
        fail("Type ID is invalid");
        break;
      case TA_BAD_ALIGNMENT:
        fail("Pointer does not align to a type");
        break;
      case TA_UNKNOWN_ADDRESS:
        fail("Address is unknown");
        break;
      default:
        fail("Unexpected error during type resolution");
    }
  };

  int actualTypeId{TA_UNKNOWN_TYPE};
  size_t actualCount{0};

  const auto get_type = [&actualTypeId, &actualCount, &ta_status](auto addr) {
    auto status = typeart_get_type(addr, &actualTypeId, &actualCount);
    if (status != TA_OK) {
      // TODO log
      ta_status(status);
    }
  };

  // typeart_resolve_type

  const auto resolve_type = [&](auto id, typeart_struct_layout& layout) {
    auto status = typeart_resolve_type(id, &layout);
    if (status != TA_OK && status != TA_WRONG_KIND) {
      // TODO log
      ta_status(status);
    }

    return status;
  };

  get_type(addr);
  if (assertk == AssertKind::STRICT) {
    if (actualTypeId != typeId) {
      type_mismatch_fail(actualTypeId);
    } else if (actualCount != count) {
      count_mismatch_fail(actualCount);
    }
  } else if (assertk == AssertKind::RELAXED) {
    if (actualTypeId != typeId) {
      bool descent = false;
      auto current_id = actualTypeId;
      do {
        typeart_struct_layout layout;
        auto status = resolve_type(current_id, layout);

        // we cannot resolve, actualTypeID is not a struct:
        if (status == TA_WRONG_KIND) {
          type_mismatch_fail(current_id);
        }

        // we have a struct, take first member id
        if (layout.count > 0) {
          current_id = layout.member_types[0];
        }
        // only continue searching if the current type ID does not match
        descent = layout.count > 0 && current_id != typeId;
      } while (descent);
    }
  }
}

inline int TYassert(int id, void* addr, size_t count, size_t typeSize, int typeId) {
  LOG_TRACE("Entering" << __FUNCTION__);
  TYdo_assert(addr, typeId, count, assert_kind);
  insert(id, {addr, typeId, count});
  impl::_do_protect(id, addr, count, typeSize, typeId);
  return 0;
}

inline int TYassert_cp() {
  LOG_TRACE("Entering" << __FUNCTION__);
  // For all stored CP data, assert again.
  for (auto [k, v] : CPs) {
    LOG_DEBUG("Checking for " << k);
    LOG_DEBUG("Addr: " << v.addr << "\ntypeId: " << v.typeId << "\nCount: " << v.count);
    __typeart_assert_type_len(v.addr, v.typeId, v.count);
  }
  return 0;
}

inline int TYdereg(int id) {
  remove(id);
  return 0;
}

}  // namespace tycart

void __tycart_assert(int id, void* addr, size_t count, size_t typeSize, int typeId) {
  LOG_TRACE("Entering" << __FUNCTION__);
  int err = tycart::TYassert(id, addr, count, typeSize, typeId);
}

void __tycart_cp_assert() {
  LOG_TRACE("Entering" << __FUNCTION__);
  tycart::TYassert_cp();
}

void __tycart_deregister_mem(int id) {
  int err = tycart::TYdereg(id);
}

void __tycart_register_FTI_t(int typeId) {
  LOG_TRACE("Entering" << __FUNCTION__);
#ifdef WITH_FTI
  tycart::impl::fti::registerFTIType(typeId);
#else
  (void)typeId;
#endif
}

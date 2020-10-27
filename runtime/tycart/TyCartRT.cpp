#include "Logger.h"
#include "tycart.h"

#include "RuntimeInterface.h"

#include <sstream>

#ifdef WITH_FTI
#include "fti.h"
#endif

#ifdef WITH_VELOC
#include "veloc.h"
#endif

#ifdef WITH_MINI_CPR
#include "mini-cpr.h"
#endif

#include <iostream>
#include <map>

namespace tycart {

enum class AssertKind { STRICT, RELAXED };

// static AssertKind assert_kind = ((std::getenv("tycart_assert") == "rel") ? AssertKind::RELAXED :
// AssertKind::RELAXED);

class TyCartRT final {
  AssertKind assert_kind{AssertKind::STRICT};

 public:
  TyCartRT() {
    const char* mode = std::getenv("tycart_assert");
    if (mode) {
      if (strcmp(mode, "rel") == 0) {
        LOG_DEBUG("Mode is set to relaxed")
        assert_kind = AssertKind::RELAXED;
      }
    }
  }

  static TyCartRT& get() {
    static TyCartRT instance;
    return instance;
  }

  AssertKind mode() const {
    return assert_kind;
  }
};

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
#ifdef WITH_FTI
namespace fti {
void registerFTIBuiltInTypes();

static std::map<int, FTIT_type> FTITs;
static bool initialized{false};

inline void registerFTIType(int typeId) {
  if (!initialized) {
    registerFTIBuiltInTypes();
    initialized = true;
  }
  FTIT_type dType;
  FTI_InitType(&dType, typeart_get_type_size(typeId));
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
inline int _do_protect(int id, void* addr, size_t count, size_t baseSize, int typeId) {
#ifdef WITH_VELOC
  (void)typeId;  // silence compiler warning
  return VELOC_Mem_protect(id, addr, count, baseSize);
#endif
#ifdef WITH_FTI
  (void)baseSize;  // silence compiler warning
  return FTI_Protect(id, addr, count, fti::getFTIType(typeId));
#endif
#ifdef WITH_MINI_CPR
  (void)typeId;
  return mini_cpr_register(id, addr, count, baseSize);
#endif
}
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

  const auto type_mismatch_fail_recurse = [&](auto actualTypeId, auto resolvedId) {
    const char* expectedName = typeart_get_type_name(typeId);
    const char* actualName = typeart_get_type_name(actualTypeId);
    const char* recursedName = typeart_get_type_name(resolvedId);
    std::stringstream ss;
    ss << "During recursive resolve: Expected type " << expectedName << "(id=" << typeId << ") but got " << actualName
       << "(id=" << actualTypeId << "). This resolved to " << recursedName;
    fail(ss.str());
  };

  const auto count_mismatch_fail = [&](auto actualCount) {
    std::stringstream ss;
    ss << "Expected number of elements is " << count << " but actual number is " << actualCount;
    fail(ss.str());
  };

  const auto count_mismatch_fail_recurse = [&](auto actualCount, auto resolvedCount) {
    std::stringstream ss;
    ss << "Expected number of elements is " << count << " resolved to  " << resolvedCount << " (from initial "
       << actualCount << ")";
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
    }
    if (actualCount != count) {
      count_mismatch_fail(actualCount);
    }
  } else if (assertk == AssertKind::RELAXED) {
    if (actualTypeId != typeId) {
      bool descent = false;
      auto current_id = actualTypeId;
      typeart_struct_layout layout;
      do {
        descent = false;

        auto status = resolve_type(current_id, layout);

        // we cannot resolve, actualTypeID is not a struct:
        if (status == TA_WRONG_KIND) {
          type_mismatch_fail_recurse(actualTypeId, current_id);
        }

        // we have a struct, take first member id
        if (layout.len > 0) {
          current_id = layout.member_types[0];
        }

        // only continue searching if the current type ID does not match
        // descent = layout.count > 0 && current_id != typeId;
        if (current_id != typeId) {
          if (layout.len == 0) {
            type_mismatch_fail_recurse(actualTypeId, current_id);
          } else {
            descent = true;
          }
        }
      } while (descent);

      // the type was resolved, but is the length as expected?
      if (count != layout.count[0]) {
        count_mismatch_fail_recurse(actualCount, layout.count[0]);
      }

    } else if (actualCount != count) {
      count_mismatch_fail(actualCount);
    }
  }
}

inline int TYassert(int id, void* addr, size_t count, size_t typeSize, int typeId) {
  LOG_TRACE("Entering" << __FUNCTION__);
  TYdo_assert(addr, typeId, count, TyCartRT::get().mode());
  insert(id, {addr, typeId, count});
  return impl::_do_protect(id, addr, count, typeSize, typeId);
}

inline int TYassert_cp() {
  LOG_TRACE("Entering" << __FUNCTION__);
  // For all stored CP data, assert again.
  for (const auto& [k, v] : CPs) {
    LOG_DEBUG("Checking for " << k);
    LOG_DEBUG("Addr: " << v.addr << "\ntypeId: " << v.typeId << "\nCount: " << v.count);
    TYdo_assert(v.addr, v.typeId, v.count, TyCartRT::get().mode());
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

void __tycart_assert_auto(int id, void* addr, size_t typeSize, int typeId) {
  const auto fail = [&](std::string msg) -> void {
    LOG_FATAL("Assert failed: " << msg);
    exit(EXIT_FAILURE);
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

  /* Query the runtime for the len information of the addr pointer and the datatype */
  int actualTypeId{TA_UNKNOWN_TYPE};
  size_t actualCount{0};

  const auto get_type = [&actualTypeId, &actualCount, &ta_status](auto addr) {
    auto status = typeart_get_type(addr, &actualTypeId, &actualCount);
    if (status != TA_OK) {
      // TODO log
      ta_status(status);
    }
  };
  get_type(addr);
  LOG_INFO("The actual count was: " << actualCount);
  __tycart_assert(id, addr, actualCount, typeSize, typeId);
  /* Output the information into some file, that the user can access later */
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

void __tycart_init(const char *cfgFile) {
#ifdef WITH_FTI
  auto res = FTI_Init(cfgFile, MPI_COMM_WORLD);
  // double check: from example https://github.com/leobago/fti/blob/master/testing/suites/features/recoverVar/checkRecoverVar.c
  if (res == FTI_NREC) {
    std::exit(res);
  }
#endif
#ifdef WITH_VELOC
  auto res = VELOC_Init(MPI_COMM_WORLD, cfgFile);
  if (res == VELOC_FAILURE) {
    std::exit(res);
  }
#endif
#ifdef WITH_MINI_CPR
  auto res = mini_cpr_init(cfgFile);
  if (res != 0) {
    std::exit(res);
  }
#endif
}

void __tycart_cp_recover(const char *name, int version) {
#ifdef WITH_FTI
  #error Currently not implemented with FTI
#endif
#ifdef WITH_VELOC
  int verCP = VELOC_Restart_test(name, version);
  if (verCP >= 0) {
    auto res = VELOC_Restart(name, verCP);
    if (res == VELOC_FAILURE) {
      std::exit(res);
    }
  }
#endif
#ifdef WITH_MINI_CPR
  int verCP = mini_cpr_restart_check(name, version);
  if (verCP >= 0) {
    auto res = mini_cpr_restart(name, verCP);
    if (res != 0) {
      std::exit(res);
    }
  }
#endif
}

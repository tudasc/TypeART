#include "Logger.h"
#include "Runtime.h"
#include "tycart.h"

#ifdef WITH_FTI
#include "fti.h"
#endif

#ifdef WITH_VELOC
#include "veloc.h"
#endif

#include <iostream>
#include <map>

namespace tycart {

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
#endif // WITH_FTI
}  // namespace impl

inline int TYassert(int id, void* addr, size_t count, size_t typeSize, int typeId) {
  LOG_TRACE("Entering" << __FUNCTION__);
  __typeart_assert_type_len(addr, typeId, count);
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

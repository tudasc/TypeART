#ifndef LIB_RUNTIME_GLOBALTYPEDEFCALLBACKS
#define LIB_RUNTIME_GLOBALTYPEDEFCALLBACKS

#include "RuntimeData.h"
#include "TypeDB.h"
#include "TypeInterface.h"
#include "support/Logger.h"
#include "typelib/TypeDatabase.h"

#include <cstdint>

namespace typeart {

class GlobalTypeTranslator final {
 private:
  RuntimeT::TypeLookupMapT translator_map;
  class Impl;
  std::unique_ptr<Impl> pImpl;

 public:
  explicit GlobalTypeTranslator(TypeDB& db);
  ~GlobalTypeTranslator();

  void register_type(const void* type);

  inline const RuntimeT::TypeLookupMapT& get_translator_map() const {
    return translator_map;
  }

  inline int get_type_id_for(MemAddr addr) const {
    return translator_map.find(addr)->second;
  }

  GlobalTypeTranslator(const GlobalTypeTranslator&)                = delete;
  GlobalTypeTranslator& operator=(const GlobalTypeTranslator&)     = delete;
  GlobalTypeTranslator(GlobalTypeTranslator&&) noexcept            = delete;
  GlobalTypeTranslator& operator=(GlobalTypeTranslator&&) noexcept = delete;
};

}  // namespace typeart

#endif /* LIB_RUNTIME_GLOBALTYPEDEFCALLBACKS */

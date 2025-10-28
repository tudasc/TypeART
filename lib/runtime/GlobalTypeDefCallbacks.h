#ifndef LIB_RUNTIME_GLOBALTYPEDEFCALLBACKS
#define LIB_RUNTIME_GLOBALTYPEDEFCALLBACKS

#include "RuntimeData.h"
#include "TypeDB.h"
#include "TypeInterface.h"
#include "support/Logger.h"
#include "typelib/TypeDatabase.h"

namespace typeart {

class GlobalTypeTranslator final {
 private:
  RuntimeT::TypeLookupMapT translator_map;
  TypeDB& type_db;
  struct GlobalTypeInfo {
    int type_id;
    const char* name;
    size_t extent;
    size_t num_members;
    const void* offsets;
    const GlobalTypeInfo* member_types;
    const void* array_sizes;
    int flag;
  };

  int struct_count{0};

  int next_type_id();

  int register_t(const GlobalTypeInfo* type);

 public:
  explicit GlobalTypeTranslator(TypeDB& db) : type_db(db) {
  }

  void register_type(const void* type);

  inline const RuntimeT::TypeLookupMapT& get_translator_map() const {
    return translator_map;
  }

  inline int get_type_id_for(MemAddr addr) const {
    return translator_map.find(addr)->second;
  }
};

}  // namespace typeart

#endif /* LIB_RUNTIME_GLOBALTYPEDEFCALLBACKS */

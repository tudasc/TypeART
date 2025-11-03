#include "GlobalTypeDefCallbacks.h"

#include "AllocationTracking.h"
#include "CallbackInterface.h"
#include "Runtime.h"
#include "RuntimeData.h"
#include "TypeInterface.h"
#include "support/Logger.h"
#include "typelib/TypeDatabase.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <sys/types.h>
#include <type_traits>
#include <vector>

namespace typeart {

#define unlikely(x)   __builtin_expect(!!(x), 0)
#define CONCAT_(x, y) x##y
#define CONCAT(x, y)  CONCAT_(x, y)
#define GUARDNAME     CONCAT(typeart_guard_, __LINE__)
#define TYPEART_RUNTIME_GUARD     \
  typeart::RTGuard GUARDNAME;     \
  if (!GUARDNAME.shouldTrack()) { \
    return;                       \
  }

struct GlobalTypeInfo {
  int type_id;
  const char* name;
  size_t extent;
  size_t num_members;
  const std::int64_t* offsets;
  const GlobalTypeInfo** member_types;
  const std::int64_t* array_sizes;
  int flag;
};

class GlobalTypeTranslator::Impl {
  TypeDatabase& type_db_;
  RuntimeT::TypeLookupMapT& translator_map_;
  int struct_count{0};
  // a fwd_decl and the decl must have the same type_id:
  RuntimeT::HashmapT<const char*, int> name_typid_fwd_decl_dedup_;

 public:
  explicit Impl(TypeDatabase& db, RuntimeT::TypeLookupMapT& translator_map) : type_db_(db), translator_map_(translator_map) {
  }

  int next_type_id(const GlobalTypeInfo* type) {
    if(auto it_type = name_typid_fwd_decl_dedup_.find(type->name); it_type != name_typid_fwd_decl_dedup_.end()){
      return it_type->second;
    }
    const int id = static_cast<int>(TYPEART_NUM_RESERVED_IDS) + struct_count;
    ++struct_count;
    name_typid_fwd_decl_dedup_.try_emplace(type->name, id);
    return id;
  }

  int register_t(const GlobalTypeInfo* type) {
    if (unlikely(type == nullptr)) {
      LOG_FATAL("Type descriptor is NULL, is it a weak extern global due to fwd decl?");
      return TYPEART_UNKNOWN_TYPE;
    }

    if (auto element = translator_map_.find(type); element != translator_map_.end()) {
      return element->second;
    }

    const bool built_in = builtins::BuiltInQuery::is_builtin_type(type->type_id);
    if (built_in) {
      translator_map_.try_emplace(type, type->type_id);
      return type->type_id;
    }

    StructTypeInfo type_descriptor;
    type_descriptor.type_id     = next_type_id(type);
    type_descriptor.name        = type->name;
    type_descriptor.extent      = type->extent;
    type_descriptor.num_members = type->num_members;
    type_descriptor.flag        = static_cast<StructTypeFlag>(type->flag);

    type_descriptor.array_sizes.reserve(type->num_members);
    type_descriptor.offsets.reserve(type->num_members);
    type_descriptor.member_types.reserve(type->num_members);
    for (auto i = 0UL; i < type->num_members; ++i) {
      const auto array_size = type->array_sizes[i];
      const auto offset     = type->offsets[i];
      type_descriptor.array_sizes.emplace_back(array_size);
      type_descriptor.offsets.emplace_back(offset);
      const auto member_id = register_t(type->member_types[i]);
      type_descriptor.member_types.emplace_back(member_id);
    }

    const bool fwd_decl = type_descriptor.flag  == StructTypeFlag::FWD_DECL;
    type_db_.registerStruct(type_descriptor, !fwd_decl);
    translator_map_.try_emplace(type, type_descriptor.type_id);

    return type_descriptor.type_id;
  }
};

GlobalTypeTranslator::GlobalTypeTranslator(TypeDatabase& db) : pImpl(std::make_unique<Impl>(db, translator_map)) {
}

GlobalTypeTranslator::~GlobalTypeTranslator() = default;

void GlobalTypeTranslator::register_type(const void* type) {
  const auto* info_struct = reinterpret_cast<const GlobalTypeInfo*>(type);
  pImpl->register_t(info_struct);
}

}  // namespace typeart

void __typeart_register_type(const void* type_ptr) {
  TYPEART_RUNTIME_GUARD;
  if (unlikely(type_ptr == nullptr)) {
    LOG_FATAL("type_ptr is NULL\n");
    return;
  }
  typeart::RuntimeSystem::get().type_translator.register_type(type_ptr);
}

#include "GlobalTypeDefCallbacks.h"

#include "CallbackInterface.h"
#include "Runtime.h"
#include "RuntimeData.h"
#include "TypeInterface.h"
#include "support/Logger.h"
#include "typelib/TypeDatabase.h"

#include <cassert>
#include <cstdint>
#include <sys/types.h>
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

class GlobalTypeTranslator::Impl {
  TypeDatabase& type_db_;
  RuntimeT::TypeLookupMapT& translator_map_;
  int struct_count{0};

 public:
  explicit Impl(TypeDatabase& db, RuntimeT::TypeLookupMapT& translator_map)
      : type_db_(db), translator_map_(translator_map) {
  }

  int next_type_id(const GlobalTypeInfo* type) {
    // a fwd_decl and the decl must have the same type_id:
    {
      const auto& struct_list = type_db_.getStructList();
      for (const auto& type_in_db : struct_list) {
        if (type_in_db.name == type->name) {
          return type_in_db.type_id;
        }
      }
    }
    const int id = static_cast<int>(TYPEART_NUM_RESERVED_IDS) + struct_count;
    ++struct_count;
    return id;
  }

  int register_t(const GlobalTypeInfo* type) {  // NOLINT(misc-no-recursion)
    if (unlikely(type == nullptr)) {
      LOG_ERROR("Type descriptor is NULL, is it a weak extern global due to fwd decl?");
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
    for (uint32_t i = 0; i < type->num_members; ++i) {
      const auto member_id  = register_t(type->member_types[i]);
      const auto array_size = type->array_sizes[i];
      const auto offset     = type->offsets[i];
      type_descriptor.array_sizes.emplace_back(array_size);
      type_descriptor.offsets.emplace_back(offset);
      type_descriptor.member_types.emplace_back(member_id);
    }

    const bool fwd_decl = type_descriptor.flag == StructTypeFlag::FWD_DECL;
    type_db_.registerStruct(type_descriptor, not fwd_decl);
    translator_map_.try_emplace(type, type_descriptor.type_id);

    return type_descriptor.type_id;
  }
};

GlobalTypeTranslator::GlobalTypeTranslator(TypeDatabase& db) : pImpl(std::make_unique<Impl>(db, translator_map)) {
}

GlobalTypeTranslator::~GlobalTypeTranslator() = default;

void GlobalTypeTranslator::register_type(const void* type) {
  const auto* info_struct = reinterpret_cast<const GlobalTypeInfo*>(type);
  const auto type_id      = pImpl->register_t(info_struct);
  LOG_DEBUG("Type id reset: " << info_struct->name << " " << info_struct->type_id << " vs. " << type_id)
  const_cast<GlobalTypeInfo*>(info_struct)->type_id = type_id;
}

}  // namespace typeart

void __typeart_register_type(const void* type_ptr) {
  TYPEART_RUNTIME_GUARD;
  if (unlikely(type_ptr == nullptr)) {
    LOG_FATAL("type_ptr is NULL\n");
    return;
  }
  typeart::RuntimeSystem::get().type_translator().register_type(type_ptr);
}

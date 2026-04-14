#include "GlobalTypeDefCallbacks.h"

#include "CallbackInterface.h"
#include "Runtime.h"
#include "RuntimeData.h"
#include "TypeInterface.h"
#include "support/Logger.h"
#include "typelib/TypeDatabase.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <sys/types.h>
#include <vector>

namespace typeart {

namespace global_types {
struct GlobalTypeInfoData {
 private:
  const char* name_;
  // Layout : [ num_member, flag, offsets...[num_member], array_sizes...[num_member] ]
  const std::uint16_t* data_;
  const GlobalTypeInfo** member_types_;

 public:
  [[nodiscard]] const char* name() const {
    assert(name_ != nullptr && "Name should not be NULL");
    return name_;
  }

  [[nodiscard]] const GlobalTypeInfo** member_types() const {
    return member_types_;
  }

  [[nodiscard]] uint16_t num_member() const {
    assert(data_ != nullptr && "Data should not be NULL");
    return data_[0];
  }

  [[nodiscard]] uint16_t flag() const {
    assert(data_ != nullptr && "Data should not be NULL");
    return data_[1];
  }

  [[nodiscard]] const uint16_t* offsets() const {
    return &data_[2];
  }

  [[nodiscard]] const uint16_t* array_sizes() const {
    return &data_[2 + num_member()];
  }
};
}  // namespace global_types

using namespace typeart::global_types;

class TypeInfoHandle {
 public:
  explicit TypeInfoHandle(const GlobalTypeInfo* info) : info_(info) {
  }

  [[nodiscard]] bool is_valid() const {
    return info_ != nullptr;
  }

  [[nodiscard]] const GlobalTypeInfo* handle() const {
    return info_;
  }

  [[nodiscard]] std::int32_t type_id() const {
    return info_->type_id;
  }

  [[nodiscard]] std::uint32_t extent() const {
    return info_->extent;
  }

  [[nodiscard]] bool is_builtin() const {
    return builtins::BuiltInQuery::is_builtin_type(type_id());
  }

  [[nodiscard]] bool has_metadata() const {
    return info_->data != nullptr;
  }

  [[nodiscard]] const char* name() const {
    return info_->data->name();
  }

  [[nodiscard]] std::uint16_t num_members() const {
    return info_->data->num_member();
  }

  [[nodiscard]] std::uint16_t flags() const {
    return info_->data->flag();
  }

  [[nodiscard]] TypeInfoHandle get_member_type(size_t index) const {
    return TypeInfoHandle(info_->data->member_types()[index]);
  }

  [[nodiscard]] std::uint16_t get_offset(size_t index) const {
    return info_->data->offsets()[index];
  }

  [[nodiscard]] std::uint16_t get_array_size(size_t index) const {
    return info_->data->array_sizes()[index];
  }

 private:
  const GlobalTypeInfo* info_;
};

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
      const auto& struct_list            = type_db_.getStructList();
      const auto* const global_type_name = type->data->name();
      for (const auto& type_in_db : struct_list) {
        if (type_in_db.name == global_type_name) {
          return type_in_db.type_id;
        }
      }
    }
    const int id = static_cast<int>(TYPEART_NUM_RESERVED_IDS) + struct_count;
    ++struct_count;
    return id;
  }

  int register_t(TypeInfoHandle type_handle) {
    if (unlikely(!type_handle.is_valid())) {
      LOG_ERROR("Type descriptor is NULL, is it a weak extern global due to fwd decl?");
      return TYPEART_UNKNOWN_TYPE;
    }

    if (auto element = translator_map_.find(type_handle.handle()); element != translator_map_.end()) {
      return element->second;
    }

    if (type_handle.is_builtin()) {
      translator_map_.try_emplace(type_handle.handle(), type_handle.type_id());
      return type_handle.type_id();
    }

    assert(type_handle.has_metadata() && "Required metadata is NULL");

    StructTypeInfo type_descriptor;
    type_descriptor.type_id     = next_type_id(type_handle.handle());
    type_descriptor.name        = type_handle.name();
    type_descriptor.extent      = type_handle.extent();
    type_descriptor.num_members = type_handle.num_members();
    type_descriptor.flag        = static_cast<StructTypeFlag>(type_handle.flags());

    const auto member_count = type_descriptor.num_members;
    type_descriptor.array_sizes.reserve(member_count);
    type_descriptor.offsets.reserve(member_count);
    type_descriptor.member_types.reserve(member_count);

    for (size_t i = 0; i < member_count; ++i) {
      const auto member_id  = register_t(type_handle.get_member_type(i));
      const auto array_size = type_handle.get_array_size(i);
      const auto offset     = type_handle.get_offset(i);

      type_descriptor.array_sizes.emplace_back(array_size);
      type_descriptor.offsets.emplace_back(offset);
      type_descriptor.member_types.emplace_back(member_id);
    }

    const bool fwd_decl = type_descriptor.flag == StructTypeFlag::FWD_DECL;
    {
      LOG_DEBUG("StructTypeInfo Dump " << (fwd_decl ? "FWD" : "") << type_descriptor.name);
      LOG_DEBUG("  Type_id: " << type_descriptor.type_id);
      LOG_DEBUG("  Extent: " << type_descriptor.extent);
      LOG_DEBUG("  Num Members: " << type_descriptor.num_members);
      LOG_DEBUG("  Flag: " << static_cast<int>(type_descriptor.flag));
      for (uint32_t i = 0; i < type_descriptor.num_members; ++i) {
        LOG_DEBUG("  Member[" << i << "]: "
                              << "ID=" << type_db_.getTypeName(type_descriptor.member_types[i]) << ", Offset="
                              << type_descriptor.offsets[i] << ", ArraySize=" << type_descriptor.array_sizes[i]);
      }
    }

    type_db_.registerStruct(type_descriptor, not fwd_decl);
    translator_map_.try_emplace(type_handle.handle(), type_descriptor.type_id);

    return type_descriptor.type_id;
  }
};

GlobalTypeTranslator::GlobalTypeTranslator(TypeDatabase& db) : pImpl(std::make_unique<Impl>(db, translator_map)) {
}

GlobalTypeTranslator::~GlobalTypeTranslator() = default;

void GlobalTypeTranslator::register_type(const void* type) {
  const auto* info_struct                           = reinterpret_cast<const GlobalTypeInfo*>(type);
  const auto type_id                                = pImpl->register_t(TypeInfoHandle{info_struct});
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

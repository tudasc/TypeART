#include "GlobalTypeDefCallbacks.h"

#include "AccessCounter.h"
#include "AllocationTracking.h"
#include "CallbackInterface.h"
#include "Runtime.h"
#include "RuntimeData.h"
#include "TypeDB.h"
#include "TypeInterface.h"
#include "support/Logger.h"
#include "typelib/TypeDatabase.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <sys/types.h>
#include <type_traits>
#include <vector>

namespace typeart {

#define TYPEART_RUNTIME_GUARD     \
  typeart::RTGuard GUARDNAME;     \
  if (!GUARDNAME.shouldTrack()) { \
    return;                       \
  }

int GlobalTypeTranslator::next_type_id() {
  const int id = static_cast<int>(TYPEART_NUM_RESERVED_IDS) + struct_count;
  ++struct_count;
  return id;
}

int GlobalTypeTranslator::register_t(const GlobalTypeInfo* type) {
  // const bool built_in = type->type_id < TYPEART_NUM_VALID_IDS;
  if (translator_map.contains(type)) {
    return translator_map[type];
  }
  builtins::BuiltInQuery query;
  const bool built_in = query.is_builtin_type(type->type_id);
  if (built_in) {
    translator_map.try_emplace(type, type->type_id);
    return type->type_id;
  }
  StructTypeInfo type_descriptor;
  type_descriptor.type_id = next_type_id();
  type_descriptor.name    = type->name;
  type_descriptor.extent  = type->extent;
  type_descriptor.flag    = static_cast<StructTypeFlag>(type->flag);

  const auto* array_sizes  = static_cast<const std::int64_t*>(type->array_sizes);
  const auto* array_offset = static_cast<const std::int64_t*>(type->offsets);
  for (auto i = 0UL; i < type->num_members; ++i) {
    type_descriptor.array_sizes.emplace_back(array_sizes[i]);
    type_descriptor.offsets.emplace_back(array_offset[i]);
    const auto member_id = register_t(static_cast<const GlobalTypeInfo*>(type->member_types));
    type_descriptor.member_types.emplace_back(member_id);
  }
  type_db.registerStruct(type_descriptor, true);
  translator_map.try_emplace(type, type_descriptor.type_id);
  return type_descriptor.type_id;
}

void GlobalTypeTranslator::register_type(const void* type) {
  const auto* info_struct = reinterpret_cast<const GlobalTypeInfo*>(type);
  LOG_TRACE(register_t(info_struct));
}

}  // namespace typeart

void __typeart_register_type(const void* type_ptr) {
  TYPEART_RUNTIME_GUARD;
  typeart::RuntimeSystem::get().type_translator.register_type(type_ptr);
}
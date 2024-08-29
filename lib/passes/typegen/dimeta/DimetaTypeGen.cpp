// TypeART library
//
// Copyright (c) 2017-2024 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#include "../TypeIDGenerator.h"
#include "Dimeta.h"
#include "DimetaData.h"
#include "support/Logger.h"
#include "typelib/TypeDB.h"
#include "typelib/TypeDatabase.h"
#include "typelib/TypeIO.h"
#include "typelib/TypeInterface.h"

#include "llvm/ADT/StringMap.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <llvm/ADT/STLExtras.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/Casting.h>
#include <memory>
#include <type_traits>

namespace llvm {
class DataLayout;
class StructType;
class Type;
class VectorType;
}  // namespace llvm

namespace typeart::types {

template <class... Ts>
struct overload : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
overload(Ts...) -> overload<Ts...>;

template <typename Type>
std::pair<std::optional<typeart_builtin_type>, int> typeid_if_ptr(const Type& type) {
  using namespace dimeta;
  const auto& quals = type.qual;
  int count{0};
  if (!quals.empty()) {
    count = llvm::count_if(quals, [](auto& qual) { return qual == Qualifier::kPtr || qual == Qualifier::kRef; });
    if (count > 0) {
      return {{TYPEART_POINTER}, count};
    }
  }
  return {{}, count};
}

template <typename Type>
size_t array_size(const Type& type) {
  using namespace dimeta;
  if constexpr (std::is_same_v<Type, typename dimeta::QualifiedCompound>) {
    return std::max(type.array_size, std::uint64_t(1));
  } else {
    return std::visit(
        overload{
            [&](const dimeta::QualifiedFundamental& f) -> size_t { return std::max(f.array_size, std::uint64_t(1)); },
            [&](const dimeta::QualifiedCompound& q) -> size_t { return std::max(q.array_size, std::uint64_t(1)); }},
        type);
  }
}

std::optional<typeart_builtin_type> get_builtin_typeid(const dimeta::QualifiedFundamental& type,
                                                       bool top_level = false) {
  const auto extent   = type.type.extent;
  const auto encoding = type.type.encoding;
  using namespace dimeta;

  const auto [ptr_type_id, count] = typeid_if_ptr(type);
  if (top_level) {
    if (count > 1) {
      LOG_FATAL("Top level, ptr ptr")
      return ptr_type_id.value();
    }
  } else {
    if (count > 0) {
      LOG_FATAL("NOT Top level, ptr ptr")
      return ptr_type_id.value();
    }
  }

  switch (encoding) {
    case FundamentalType::Encoding::kUnknown:
      return TYPEART_UNKNOWN_TYPE;
    case FundamentalType::Encoding::kVoid:
      return TYPEART_INT8;
    case FundamentalType::Encoding::kChar:
    case FundamentalType::Encoding::kSignedChar:
    case FundamentalType::Encoding::kUnsignedChar:
    case FundamentalType::Encoding::kBool:
    case FundamentalType::Encoding::kSignedInt:
    case FundamentalType::Encoding::kUnsignedInt: {
      switch (extent) {
        case 4:
          return TYPEART_INT32;
        case 8:
          return TYPEART_INT64;
        case 2:
          return TYPEART_INT16;
        case 1:
          return TYPEART_INT8;
        default:
          return TYPEART_UNKNOWN_TYPE;
      }
    }
    case FundamentalType::Encoding::kFloat: {
      switch (extent) {
        case 4:
          return TYPEART_FLOAT;
        case 8:
          return TYPEART_DOUBLE;
        case 2:
          return TYPEART_HALF;
        case 16:
          return TYPEART_FP128;
        default:
          return TYPEART_UNKNOWN_TYPE;
      }
    }
    default:
      break;
  }
  return TYPEART_UNKNOWN_TYPE;
}

class DimetaTypeManager final : public TypeIDGenerator {
 public:
  explicit DimetaTypeManager(std::string file_) : TypeIDGenerator(std::move(file_)) {
  }

  int getOrRegister(const dimeta::QualifiedType& type, bool top_level = false) {
    const auto fetch_id = [&](const auto name) -> std::optional<int> {
      if (auto it = structMap.find(name); it != structMap.end()) {
        const auto type_id = it->second;
        if (!typeDB.isUserDefinedType(type_id)) {
          LOG_ERROR("Expected user defined struct type " << name << " for type id: " << type_id);
          return TYPEART_UNKNOWN_TYPE;
        }
        return type_id;
      }
      return {};
    };

    auto type_id =
        std::visit(overload{[&](const dimeta::QualifiedFundamental& f) -> int {
                              LOG_FATAL("QualFunda " << f.type.name)
                              if (f.is_vector) {
                                assert(!f.typedef_name.empty() && "Vector types need to be typedef'ed for now!");

                                const auto vec_name    = f.typedef_name;
                                const auto existing_id = fetch_id(vec_name);
                                if (existing_id) {
                                  return existing_id.value();
                                }

                                const int id = reserveNextTypeId();
                                StructTypeInfo struct_info;
                                struct_info.type_id = id;
                                struct_info.name    = vec_name;

                                struct_info.extent = f.type.extent * std::max<int>(1, f.array_size);
                                // FIXME assume vector offsets are "packed":
                                for (int i = 0; i < f.array_size; ++i) {
                                  struct_info.offsets.push_back(i * f.type.extent);
                                  struct_info.array_sizes.push_back(1);
                                }

                                struct_info.num_members = f.array_size;
                                struct_info.flag        = StructTypeFlag::LLVM_VECTOR;

                                return id;
                              }
                              return get_builtin_typeid(f, top_level).value();
                            },
                            [&](const dimeta::QualifiedCompound& q) -> int {
                              LOG_FATAL("QualCompound " << q.type.name)
                              const auto& compound = q.type;
                              const auto name      = compound.identifier.empty() ? compound.name : compound.identifier;

                              const auto [ptr_type_id, count] = typeid_if_ptr(q);
                              if (top_level) {
                                if (count > 1) {
                                  return ptr_type_id.value();
                                }
                              } else {
                                if (count > 0) {
                                  return ptr_type_id.value();
                                }
                              }

                              const auto existing_id = fetch_id(name);
                              if (existing_id) {
                                return existing_id.value();
                              }

                              const int id = reserveNextTypeId();
                              StructTypeInfo struct_info;
                              struct_info.type_id = id;
                              struct_info.name    = name;

                              struct_info.extent = compound.extent * std::max<int>(1, q.array_size);
                              LOG_FATAL(compound.extent << " " << std::max<int>(1, q.array_size))
                              struct_info.offsets     = compound.offsets;
                              //   struct_info.array_sizes = compound.sizes;
                              struct_info.num_members = compound.bases.size() + compound.members.size();
                              struct_info.flag        = StructTypeFlag::USER_DEFINED;

                              for (const auto& base : compound.bases) {
                                struct_info.member_types.push_back(getOrRegister(base->base));
                                struct_info.array_sizes.push_back(array_size(base->base));
                              }
                              for (const auto& member : compound.members) {
                                struct_info.member_types.push_back(getOrRegister(member->member));
                                struct_info.array_sizes.push_back(array_size(member->member));
                              }

                              typeDB.registerStruct(struct_info);

                              structMap.insert({struct_info.name, id});

                              return id;
                            }},
                   type);

    LOG_FATAL("Returning " << type_id);
    return type_id;
  }

  [[nodiscard]] int getOrRegisterType(llvm::Value* type) {
    if (auto call = llvm::dyn_cast<llvm::CallBase>(type)) {
      auto val = dimeta::located_type_for(call);

      if (val) {
        LOG_FATAL("Registering")

        return getOrRegister(val->type, true);
      }
    } else if (auto alloc = llvm::dyn_cast<llvm::AllocaInst>(type)) {
      auto val = dimeta::located_type_for(alloc);
      if (val) {
        LOG_FATAL("Registering alloca")

        return getOrRegister(val->type, true);
      }
    } else if (auto global = llvm::dyn_cast<llvm::GlobalVariable>(type)) {
      auto val = dimeta::located_type_for(global);
      if (val) {
        LOG_FATAL("Registering global")

        return getOrRegister(val->type, true);
      }
    }
    return TYPEART_UNKNOWN_TYPE;
  }

  TypeIdentifier getOrRegisterType(const MallocData& data) override {
    return {getOrRegisterType(data.call), 0};
  }

  TypeIdentifier getOrRegisterType(const AllocaData& data) override {
    return {getOrRegisterType(data.alloca), 0};
  }

  TypeIdentifier getOrRegisterType(const GlobalData& data) override {
    return {getOrRegisterType(data.global), 0};
  }

  ~DimetaTypeManager() = default;
};

std::unique_ptr<typeart::TypeGenerator> make_dimeta_typeidgen(std::string_view file) {
  return std::make_unique<typeart::types::DimetaTypeManager>(std::string{file});
}

}  // namespace typeart::types

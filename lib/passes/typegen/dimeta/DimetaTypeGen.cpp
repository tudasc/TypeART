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
#include "typegen/TypeGenerator.h"
#include "typelib/TypeDatabase.h"
#include "typelib/TypeInterface.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <llvm/ADT/STLExtras.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/Casting.h>
#include <memory>
#include <string>
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
dimeta::ArraySize array_size(const Type& type) {
  using namespace dimeta;

  if constexpr (std::is_same_v<Type, typename dimeta::QualifiedCompound> ||
                std::is_same_v<Type, typename dimeta::QualifiedFundamental>) {
    if (type.is_vector) {
      return std::uint64_t(1);
    }
    const auto array_size_factor = type.array_size.empty() ? 1 : type.array_size.at(0);
    LOG_FATAL(array_size_factor)
    return array_size_factor;
  } else {
    return std::visit(overload{[&](const dimeta::QualifiedFundamental& f) -> size_t { return array_size(f); },
                               [&](const dimeta::QualifiedCompound& q) -> size_t { return array_size(q); }},
                      type);
  }
}

template <typename Type>
std::string name_or_typedef_of(const Type& type) {
  using namespace dimeta;

  if constexpr (std::is_same_v<Type, typename dimeta::QualifiedCompound> ||
                std::is_same_v<Type, typename dimeta::QualifiedFundamental>) {
    return type.type.name.empty() ? type.typedef_name : type.type.name;
  } else {
    return std::visit(overload{[&](const dimeta::QualifiedFundamental& f) { return name_or_typedef_of(f); },
                               [&](const dimeta::QualifiedCompound& q) { return name_or_typedef_of(q); }},
                      type);
  }
}

std::optional<typeart_builtin_type> get_builtin_typeid(const dimeta::QualifiedFundamental& type,
                                                       bool top_level = false) {
  const auto extent   = type.type.extent;
  const auto encoding = type.type.encoding;
  using namespace dimeta;

  const auto [ptr_type_id, count] = typeid_if_ptr(type);

  // const bool root_pointer_alloc = top_level && count > 1;
  // const bool pointer_alloc      = !top_level && count > 0;
  // if (root_pointer_alloc || pointer_alloc) {
  //   LOG_FATAL((top_level ? "Top level, ptr ptr" : "NOT Top level, ptr ptr"));
  //   return ptr_type_id.value();
  // }

  if (top_level) {
    if (count > 1) {
      LOG_FATAL("Top level, ptr ptr")
      return ptr_type_id.value();
    }
  } else if (count > 0) {
    LOG_FATAL("NOT Top level, ptr ptr")
    return ptr_type_id.value();
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
  // explicit DimetaTypeManager(std::string file_) : TypeIDGenerator(std::move(file_)) {
  // }

  using TypeIDGenerator::TypeIDGenerator;

  int getOrRegister(const dimeta::QualifiedType& type, bool top_level = false) {
    const auto fetch_id = [&](const auto name) -> std::optional<int> {
      if (auto it = structMap.find(name); it != structMap.end()) {
        const auto type_id = it->second;
        // if (!typeDB->isUserDefinedType(type_id) && !typeDB->isVectorType(type_id)) {
        // LOG_ERROR("Expected user defined struct/vector type " << name << " for type id: " << type_id);
        // return TYPEART_UNKNOWN_TYPE;
        // }
        return type_id;
      }
      return {};
    };

    auto type_id =
        std::visit(overload{[&](const dimeta::QualifiedFundamental& f) -> int {
                              LOG_FATAL("QualFunda " << f.type.name)
                              if (f.is_vector) {
                                LOG_FATAL("Vec type found")
                                assert(!f.typedef_name.empty() && "Vector types need to be typedef'ed for now!");

                                const auto vec_name    = f.typedef_name;
                                const auto existing_id = fetch_id(vec_name);
                                if (existing_id) {
                                  LOG_FATAL("Vec type found with id " << existing_id.value())
                                  return existing_id.value();
                                }

                                const int id = reserveNextTypeId();
                                StructTypeInfo struct_info;
                                struct_info.type_id          = id;
                                struct_info.name             = vec_name;
                                const auto array_size_factor = array_size(f);
                                struct_info.extent           = f.type.extent * array_size_factor;

                                const auto vec_member_id = get_builtin_typeid(f, top_level).value();
                                // FIXME assume vector offsets are "packed":
                                for (std::uint64_t i = 0; i < array_size_factor; ++i) {
                                  struct_info.offsets.push_back(i * f.type.extent);
                                  struct_info.array_sizes.push_back(1);
                                  struct_info.member_types.push_back(vec_member_id);
                                }

                                struct_info.num_members = array_size_factor;
                                struct_info.flag        = StructTypeFlag::LLVM_VECTOR;

                                LOG_FATAL("Registered Vec type found with id " << id)
                                typeDB->registerStruct(struct_info);
                                structMap.insert({struct_info.name, id});

                                return id;
                              }
                              return get_builtin_typeid(f, top_level).value();
                            },
                            [&](const dimeta::QualifiedCompound& q) -> int {
                              const std::string name_or_typedef = name_or_typedef_of(q);
                              LOG_FATAL("QualCompound \"" << name_or_typedef << "\"") using namespace dimeta;
                              if (q.type.name.empty() && q.type.type == CompoundType::Tag::kUnknown) {
                                LOG_FATAL("Potentially pointer to (member) function, skipping.")
                                return TYPEART_UNKNOWN_TYPE;
                              }

                              const auto& compound            = q.type;
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

                              bool is_forward_declaration = false;
                              const auto existing_id      = fetch_id(name_or_typedef);
                              if (existing_id) {
                                const auto* struct_info = typeDB->getStructInfo(existing_id.value());
                                if (struct_info->flag == StructTypeFlag::FWD_DECL) {
                                  is_forward_declaration = true;
                                } else {
                                  return existing_id.value();
                                }
                              }

                              const int id = is_forward_declaration ? existing_id.value() : reserveNextTypeId();
                              StructTypeInfo struct_info;
                              struct_info.type_id = id;
                              struct_info.name    = name_or_typedef;
                              if (q.is_forward_decl) {
                                struct_info.flag = StructTypeFlag::FWD_DECL;
                              } else {
                                struct_info.flag = StructTypeFlag::USER_DEFINED;
                              }

                              const auto array_size_factor = array_size(q);
                              struct_info.extent           = compound.extent * array_size_factor;
                              LOG_FATAL(compound.extent << " " << array_size_factor)
                              struct_info.offsets     = compound.offsets;
                              //   struct_info.array_sizes = compound.sizes;
                              struct_info.num_members = compound.bases.size() + compound.members.size();

                              for (const auto& base : compound.bases) {
                                struct_info.member_types.push_back(getOrRegister(base->base));
                                struct_info.array_sizes.push_back(array_size(base->base));
                              }
                              for (const auto& member : compound.members) {
                                struct_info.member_types.push_back(getOrRegister(member->member));
                                struct_info.array_sizes.push_back(array_size(member->member));
                              }

                              typeDB->registerStruct(struct_info, is_forward_declaration);

                              structMap.insert({struct_info.name, id});

                              return id;
                            }},
                   type);

    LOG_FATAL("Returning " << type_id);
    return type_id;
  }

  [[nodiscard]] TypeIdentifier getOrRegisterTypeValue(llvm::Value* type) {
    if (auto call = llvm::dyn_cast<llvm::CallBase>(type)) {
      // LOG_FATAL(*type)
      auto val = dimeta::located_type_for(call);

      if (val) {
        LOG_FATAL("Registering")

        return {getOrRegister(val->type, true), array_size(val->type)};
      }
    } else if (auto alloc = llvm::dyn_cast<llvm::AllocaInst>(type)) {
      auto val = dimeta::located_type_for(alloc);
      if (val) {
        LOG_FATAL("Registering alloca")
        const auto type_id        = getOrRegister(val->type, true);
        const auto array_size_val = array_size(val->type);
        LOG_FATAL(array_size_val)
        return {type_id, array_size_val};
      }
    } else if (auto global = llvm::dyn_cast<llvm::GlobalVariable>(type)) {
      auto val = dimeta::located_type_for(global);
      if (val) {
        LOG_FATAL("Registering global")

        return {getOrRegister(val->type, true), array_size(val->type)};
      }
    }
    return {TYPEART_UNKNOWN_TYPE, 0};
  }

  void registerModule(const ModuleData& module) override {
    using namespace dimeta;
    // std::optional<CompileUnitTypeList> compile_unit_types(const llvm::Module*)
    LOG_FATAL("Register module types")
    auto cu_types_list = dimeta::compile_unit_types(module.module).value_or(dimeta::CompileUnitTypeList{});

    for (const auto& cu : cu_types_list) {
      const QualifiedTypeList& list = cu.types;
      for (const auto& cu_type : list) {
        getOrRegister(cu_type);
      }
    }
    LOG_FATAL("Done: Register module types")
  }

  TypeIdentifier getOrRegisterType(const MallocData& data) override {
    return getOrRegisterTypeValue(data.call);
  }

  TypeIdentifier getOrRegisterType(const AllocaData& data) override {
    LOG_FATAL("Start register alloca \"" << *data.alloca << "\"")
    const auto alloc_type = getOrRegisterTypeValue(data.alloca);
    return {alloc_type.type_id, alloc_type.num_elements};
  }

  TypeIdentifier getOrRegisterType(const GlobalData& data) override {
    return getOrRegisterTypeValue(data.global);
  }

  ~DimetaTypeManager() = default;
};

std::unique_ptr<typeart::TypeGenerator> make_dimeta_typeidgen(std::string_view file,
                                                              std::unique_ptr<TypeDatabase> database_of_types) {
  return std::make_unique<typeart::types::DimetaTypeManager>(std::string{file}, std::move(database_of_types));
}

}  // namespace typeart::types

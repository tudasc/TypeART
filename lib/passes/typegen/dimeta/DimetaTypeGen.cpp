// TypeART library
//
// Copyright (c) 2017-2025 TypeART Authors
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

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MD5.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
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
  int count = llvm::count_if(quals, [](auto& qual) { return qual == Qualifier::kPtr || qual == Qualifier::kRef; });

  if constexpr (std::is_same_v<Type, typename dimeta::QualifiedFundamental>) {
    if (type.type.encoding == FundamentalType::Encoding::kVtablePtr) {
      return {TYPEART_VTABLE_POINTER, count};
    }
  }
  if (count > 0) {
    return {{TYPEART_POINTER}, count};
  }

  return {{}, count};
}

namespace detail {

template <typename Type, typename Func>
auto apply_function(const Type& type, Func&& handle_qualified_type) {
  using namespace dimeta;

  if constexpr (std::is_same_v<Type, typename dimeta::QualifiedCompound> ||
                std::is_same_v<Type, typename dimeta::QualifiedFundamental>) {
    return std::forward<Func>(handle_qualified_type)(type);
  } else {
    return std::visit(
        [&](auto&& qualified_type) {
          return apply_function(qualified_type, std::forward<Func>(handle_qualified_type));
        },
        type);
  }
}

}  // namespace detail

namespace workaround {
void remove_pointer_level(const llvm::AllocaInst* alloc, dimeta::LocatedType& val) {
  // If the alloca instruction is not a pointer, but the located_type has a pointer-like qualifier, we remove it.
  // Workaround for inlining issue, see test typemapping/05_milc_inline_metadata.c
  // TODO Should be removed if dimeta fixes it.
  if (!alloc->getAllocatedType()->isPointerTy()) {
    LOG_DEBUG("Alloca is not a pointer")

    const auto remove_pointer_level = [](auto& qual) {
      auto pointer_like_iter = llvm::find_if(qual, [](auto qualifier) {
        switch (qualifier) {
          case dimeta::Qualifier::kPtr:
          case dimeta::Qualifier::kRef:
          case dimeta::Qualifier::kPtrToMember:
            return true;
          default:
            break;
        }
        return false;
      });
      if (pointer_like_iter != std::end(qual)) {
        LOG_DEBUG("Removing pointer level " << static_cast<int>(*pointer_like_iter))
        qual.erase(pointer_like_iter);
      }
    };
    std::visit([&](auto&& qualified_type) { remove_pointer_level(qualified_type.qual); }, val.type);
  }
}
}  // namespace workaround

template <typename Type>
dimeta::ArraySize vector_num_elements(const Type& type) {
  return detail::apply_function(type, [](const auto& t) -> dimeta::Extent {
    if (t.is_vector) {
      int pos{-1};
      // Find kVector tag-position to determine vector size
      for (const auto& qualifier : t.qual) {
        if (qualifier == dimeta::Qualifier::kVector) {
          pos++;
          break;
        }
        if (qualifier == dimeta::Qualifier::kArray) {
          pos++;
        }
      }
      if (pos == -1) {
        return 1;
      }
      return t.array_size.at(pos);
    }

    return 1;
  });
}

std::string get_anon_struct_identifier(const dimeta::QualifiedCompound& compound) {
  llvm::MD5 compound_hash;
  if (compound.type.members.empty()) {
    LOG_WARNING("Anonymous struct has no members")
  }
  for (const auto& [member, offset, size] :
       llvm::zip(compound.type.members, compound.type.offsets, compound.type.sizes)) {
    compound_hash.update(member->name);
    compound_hash.update(offset);
    compound_hash.update(size);
    compound_hash.update(std::visit(overload{[&](const dimeta::QualifiedFundamental& member_fundamental) {
                                               return std::to_string(
                                                          static_cast<int>(member_fundamental.type.encoding)) +
                                                      std::to_string(static_cast<int>(member_fundamental.type.extent));
                                             },
                                             [&](const dimeta::QualifiedCompound& member_compound) {
                                               return get_anon_struct_identifier(member_compound);
                                             }},
                                    member->member));
    compound_hash.update("\0");
  }
  compound_hash.update(compound.type.extent);
  compound_hash.update("\0");
  llvm::MD5::MD5Result hash_result;
  compound_hash.final(hash_result);
  return "anonymous_compound_" + std::string(hash_result.digest().str());
}

template <typename Type>
dimeta::ArraySize array_size(const Type& type) {
  return detail::apply_function(type, [](const auto& t) -> dimeta::Extent {
    if (t.array_size.size() > 1 || (t.is_vector && t.array_size.size() > 2)) {
      LOG_ERROR("Unsupported array size number count > 1 for array type or > 2 for vector")
    }
    // Vector array-size does not count towards array type-size
    if (t.is_vector && t.array_size.size() == 1) {
      return 1;
    }
    const auto array_size_factor = t.array_size.empty() ? 1 : t.array_size.at(0);

    return array_size_factor;
  });
}

template <typename Type>
std::string name_or_typedef_of(const Type& type) {
  return detail::apply_function(type, [](const auto& qual_type) {
    const bool no_name = qual_type.type.name.empty();
    if constexpr (std::is_same_v<Type, typename dimeta::QualifiedCompound>) {
      const bool no_identifier = qual_type.type.identifier.empty();
      const bool no_typedef    = qual_type.typedef_name.empty();
      if (no_identifier && no_name && no_typedef) {
        return get_anon_struct_identifier(qual_type);
      }
      if (no_identifier && no_name) {
        return qual_type.typedef_name;
      }
      if (no_identifier) {
        return qual_type.type.name;
      }
      return qual_type.type.identifier;
    }

    return no_name ? qual_type.typedef_name : qual_type.type.name;
  });
}

std::optional<typeart_builtin_type> get_builtin_typeid(const dimeta::QualifiedFundamental& type,
                                                       bool top_level = false) {
  using namespace dimeta;
  const auto [ptr_type_id, count] = typeid_if_ptr(type);

  if ((top_level && count > 1) || (!top_level && count > 0)) {
    LOG_DEBUG((top_level ? "Top level, ptr ptr" : "NOT Top level, ptr ptr"));
    return ptr_type_id.value();
  }

  const auto extent   = type.type.extent;
  const auto encoding = type.type.encoding;

  switch (encoding) {
    case FundamentalType::Encoding::kVtablePtr:
      return TYPEART_VTABLE_POINTER;
    case FundamentalType::Encoding::kUnknown:
      return TYPEART_UNKNOWN_TYPE;
    case FundamentalType::Encoding::kVoid:
      return TYPEART_VOID;
    case FundamentalType::kNullptr:
      return TYPEART_NULLPOINTER;
    case FundamentalType::Encoding::kUTFChar: {
      switch (extent) {
        case 4:
          return TYPEART_UTF_CHAR_32;
        case 2:
          return TYPEART_UTF_CHAR_16;
        case 1:
          return TYPEART_UTF_CHAR_8;
        default:
          return TYPEART_UNKNOWN_TYPE;
      }
    }
    case FundamentalType::Encoding::kChar:
    case FundamentalType::Encoding::kSignedChar:
      return TYPEART_CHAR_8;
    case FundamentalType::Encoding::kUnsignedChar:
      return TYPEART_UCHAR_8;
    case FundamentalType::Encoding::kBool:
      return TYPEART_BOOL;
    case FundamentalType::Encoding::kUnsignedInt: {
      switch (extent) {
        case 4:
          return TYPEART_UINT_32;
        case 8:
          return TYPEART_UINT_64;
        case 2:
          return TYPEART_UINT_16;
        case 1:
          return TYPEART_UINT_8;
        case 16:
          return TYPEART_UINT_128;
        default:
          return TYPEART_UNKNOWN_TYPE;
      }
    }
    case FundamentalType::Encoding::kSignedInt: {
      switch (extent) {
        case 4:
          return TYPEART_INT_32;
        case 8:
          return TYPEART_INT_64;
        case 2:
          return TYPEART_INT_16;
        case 1:
          return TYPEART_INT_8;
        case 16:
          return TYPEART_INT_128;
        default:
          return TYPEART_UNKNOWN_TYPE;
      }
    }
    case FundamentalType::Encoding::kComplex: {
      switch (extent) {
        case 2:
          return TYPEART_COMPLEX_64;
        case 4:
          return TYPEART_COMPLEX_128;
        case 1:
          return TYPEART_COMPLEX_256;
        default:
          return TYPEART_UNKNOWN_TYPE;
      }
    }
    case FundamentalType::Encoding::kFloat: {
      switch (extent) {
        case 4:
          return TYPEART_FLOAT_32;
        case 8:
          return TYPEART_FLOAT_64;
        case 2:
          return TYPEART_FLOAT_16;
        case 16:
          return TYPEART_FLOAT_128;
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

  std::optional<int> fetch_id(const std::string& name) {
    if (auto it = structMap.find(name); it != structMap.end()) {
      const auto type_id = it->second;
      return type_id;
    }
    return {};
  }

  int registerVectorType(const dimeta::QualifiedFundamental& f, bool top_level) {
    LOG_DEBUG("Vector-like type found")
    assert(!f.typedef_name.empty() && "Vector types need to be typedef'ed.");

    const auto vec_name    = f.typedef_name;
    const auto existing_id = fetch_id(vec_name);
    if (existing_id) {
      LOG_DEBUG("Registered type found with id " << existing_id.value())
      return existing_id.value();
    }

    const int id = reserveNextTypeId();
    StructTypeInfo struct_info;
    struct_info.type_id     = id;
    struct_info.name        = vec_name;
    const auto num_elements = vector_num_elements(f);
    struct_info.extent      = f.vector_size;  // f.type.extent * array_size_factor;

    const auto vec_member_id = get_builtin_typeid(f, top_level).value();
    // FIXME assume vector offsets are "packed":
    for (std::uint64_t i = 0; i < num_elements; ++i) {
      struct_info.offsets.push_back(i * f.type.extent);
      struct_info.array_sizes.push_back(1);
      struct_info.member_types.push_back(vec_member_id);
    }

    struct_info.num_members = num_elements;
    struct_info.flag        = StructTypeFlag::LLVM_VECTOR;

    LOG_DEBUG("Registering vector-like type with id " << id)
    typeDB->registerStruct(struct_info);
    structMap.insert({struct_info.name, id});

    return id;
  }

  int registerStructType(const dimeta::QualifiedCompound& q, bool top_level) {
    const std::string name_or_typedef = name_or_typedef_of(q);
    LOG_DEBUG("QualifiedCompound \"" << name_or_typedef << "\"");
    using namespace dimeta;
    if (q.type.name.empty() && q.type.type == CompoundType::Tag::kUnknown) {
      LOG_DEBUG("Potentially pointer to (member) function, skipping.")
      return TYPEART_UNKNOWN_TYPE;
    }

    const auto& compound            = q.type;
    const auto [ptr_type_id, count] = typeid_if_ptr(q);

    if ((top_level && count > 1) || (!top_level && count > 0)) {
      // First: if top level allocation then the first pointer can be ignored, but
      // the second is not. Second: if it's not a top level allocation, we assume
      // it's a pointer type
      return ptr_type_id.value();
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

    struct_info.extent = compound.extent;

    size_t num_bases{0};
    for (const auto& base : compound.bases) {
      if (base->is_empty_base_class) {
        continue;
      }
      num_bases++;
      struct_info.member_types.push_back(getOrRegister(base->base));
      struct_info.array_sizes.push_back(array_size(base->base));
      struct_info.offsets.push_back(base->offset);
    }

    struct_info.num_members = compound.members.size() + num_bases;
    struct_info.offsets.insert(std::end(struct_info.offsets),  //
                               std::begin(compound.offsets),   //
                               std::end(compound.offsets));

    for (const auto& member : compound.members) {
      struct_info.member_types.push_back(getOrRegister(member->member));
      struct_info.array_sizes.push_back(array_size(member->member));
    }

    LOG_DEBUG("Registering struct-like type with id " << id)
    typeDB->registerStruct(struct_info, is_forward_declaration);
    structMap.insert({struct_info.name, id});

    return id;
  }

  int getOrRegister(const dimeta::QualifiedType& type, bool top_level = false) {
    auto type_id = std::visit(
        overload{[&](const dimeta::QualifiedFundamental& f) -> int {
                   LOG_DEBUG("QualifiedFundamental \"" << f.type.name << "\"");
                   if (f.is_vector) {
                     return registerVectorType(f, top_level);
                   }
                   return get_builtin_typeid(f, top_level).value();
                 },
                 [&](const dimeta::QualifiedCompound& q) -> int { return registerStructType(q, top_level); }},
        type);

    LOG_DEBUG("Returning type-id " << type_id);
    return type_id;
  }

  [[nodiscard]] TypeIdentifier getOrRegisterTypeValue(llvm::Value* type) {
    if (auto call = llvm::dyn_cast<llvm::CallBase>(type)) {
      // LOG_DEBUG(*type)
      auto val = dimeta::located_type_for(call);

      if (val) {
        LOG_DEBUG("Registering malloc-like")

        return {getOrRegister(val->type, true), array_size(val->type)};
      }
    } else if (auto* alloc = llvm::dyn_cast<llvm::AllocaInst>(type)) {
      LOG_DEBUG("Alloca found")
      auto val = dimeta::located_type_for(alloc);
      if (val) {
        LOG_DEBUG("Registering alloca")
        workaround::remove_pointer_level(alloc, val.value());
        const auto type_id        = getOrRegister(val->type, false);
        const auto array_size_val = array_size(val->type);
        LOG_DEBUG(array_size_val)
        return {type_id, array_size_val};
      }
    } else if (auto* global = llvm::dyn_cast<llvm::GlobalVariable>(type)) {
      auto val = dimeta::located_type_for(global);
      if (val) {
        LOG_DEBUG("Registering global")

        return {getOrRegister(val->type, true), array_size(val->type)};
      }
    }
    return {TYPEART_UNKNOWN_TYPE, 0};
  }

  void registerModule(const ModuleData& module) override {
    using namespace dimeta;
    // std::optional<CompileUnitTypeList> compile_unit_types(const llvm::Module*)
    LOG_DEBUG("Register module types")
    auto cu_types_list = dimeta::compile_unit_types(module.module).value_or(dimeta::CompileUnitTypeList{});

    for (const auto& cu : cu_types_list) {
      const QualifiedTypeList& list = cu.types;
      for (const auto& cu_type : list) {
        getOrRegister(cu_type);
      }
    }
    LOG_DEBUG("Done: Register module types")
  }

  TypeIdentifier getOrRegisterType(const MallocData& data) override {
    // LOG_INFO("Start register malloc-like \"" << *data.call << "\"")
    return getOrRegisterTypeValue(data.call);
  }

  TypeIdentifier getOrRegisterType(const AllocaData& data) override {
    // LOG_INFO("Start register alloca \"" << *data.alloca << "\"")
    const auto alloc_type = getOrRegisterTypeValue(data.alloca);
    return {alloc_type.type_id, alloc_type.num_elements};
  }

  TypeIdentifier getOrRegisterType(const GlobalData& data) override {
    // LOG_INFO("Start register global \"" << *data.global << IR"\"")
    return getOrRegisterTypeValue(data.global);
  }

  ~DimetaTypeManager() = default;
};

std::unique_ptr<typeart::TypeGenerator> make_dimeta_typeidgen(std::string_view file,
                                                              std::unique_ptr<TypeDatabase> database_of_types) {
  return std::make_unique<typeart::types::DimetaTypeManager>(std::string{file}, std::move(database_of_types));
}

}  // namespace typeart::types

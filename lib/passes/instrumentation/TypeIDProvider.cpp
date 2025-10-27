#include "TypeIDProvider.h"

#include "TypeDatabase.h"
#include "TypeInterface.h"
#include "configuration/Configuration.h"
#include "support/ConfigurationBase.h"
#include "support/Logger.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace typeart {

void TypeRegistry::registerModule(const ModuleData&) {
}

class TypeRegistryNoOp final : public TypeRegistry {
 public:
  llvm::Value* getOrRegister(llvm::Value* type_id_const) override {
    return type_id_const;
  }
};

namespace helper {
inline int get_type_id(llvm::Value* type_id_const) {
  auto* constant_int = llvm::dyn_cast<llvm::ConstantInt>(type_id_const);
  assert(constant_int && "Expected llvm::ConstantInt");
  if (!constant_int) {
    return TYPEART_UNKNOWN_TYPE;
  }
  const int type_id = static_cast<int>(constant_int->getSExtValue());
  return type_id;
}

template <typename... Args>
inline std::string concat(Args&&... args) {
  const auto str = (llvm::StringRef{} + ... + llvm::StringRef{std::forward<Args>(args)});
  return str.str();
}

template <typename... Args>
inline std::string create_prefixed_name(Args&&... args) {
  return concat(llvm::StringRef{"_typeart_"}, std::forward<Args>(args)...);
}

}  // namespace helper

namespace typedb {

struct GlobalTypeData {
  struct TypeData {
    llvm::Constant* type_struct;
    llvm::GlobalVariable* type;
    llvm::Constant* name;
    llvm::Constant* offset;
    llvm::Constant* count;
  };
  llvm::StringMap<TypeData> global_type_data;

  inline bool has_type_name(llvm::StringRef name) const {
    return global_type_data.contains(name);
  }

  inline const TypeData& get_type(llvm::StringRef name) const {
    return global_type_data.at(name);
  }
};

struct GlobalTypeRegistrar {
  llvm::Module* module_;
  const TypeDatabase* type_db_;
  llvm::IRBuilder<> ir_build;
  llvm::StructType* struct_layout_type_;
  GlobalTypeData global_types_;

  GlobalTypeRegistrar(llvm::Module* m, const TypeDatabase* type_db)
      : module_(m), type_db_(type_db), ir_build(m->getContext()) {
    declareLayout();
  }

 private:
  void declareLayout() {
    auto& context = module_->getContext();
    llvm::IRBuilder<> Builder(context);
    struct_layout_type_ = llvm::StructType::create(context, "struct.typeart_struct_layout_t");
    struct_layout_type_->setBody({
        Builder.getInt32Ty(),                   // int type_id
        llvm::PointerType::getUnqual(context),  // const char* name
        Builder.getInt64Ty(),                   // size_t extent
        Builder.getInt64Ty(),                   // size_t num_members
        llvm::PointerType::getUnqual(context),  // const size_t* offsets
        llvm::PointerType::getUnqual(context),  // const typeart_struct_layout_t* member_types
        llvm::PointerType::getUnqual(context),  // const size_t* count
        Builder.getInt32Ty(),                   // int type_flag
    });
  }

  llvm::GlobalVariable* create_global(
      llvm::StringRef name, llvm::Type* type, llvm::Constant* init = nullptr,
      llvm::GlobalVariable::LinkageTypes link_type = llvm::GlobalValue::WeakODRLinkage) {
    // TODO: https://llvm.org/docs/LangRef.html#linkage w.r.t. forward declared types
    auto* global_struct =
        new llvm::GlobalVariable(*module_, type, true, link_type, init, helper::create_prefixed_name(name));
    return global_struct;
  }

  llvm::Constant* create_global_constant_string(llvm::StringRef name) {
    // TODO think about linkage
    // auto* name_str = ir_build.CreateGlobalStringPtr(name, helper::create_prefixed_name("typename_", name), 0,
    // module_);
    auto* global_string =
        ir_build.CreateGlobalString(name, helper::create_prefixed_name("typename_", name), 0, module_);
    global_string->setConstant(true);
    global_string->setLinkage(llvm::GlobalValue::WeakODRLinkage);
    auto* i32_zero = llvm::ConstantInt::get(llvm::Type::getInt32Ty(module_->getContext()), 0);
    return llvm::ConstantExpr::getInBoundsGetElementPtr(global_string->getValueType(), global_string,
                                                        llvm::ArrayRef<llvm::Constant*>{i32_zero, i32_zero});
  }

  llvm::Constant* create_global_array_ptr(const llvm::StringRef name, llvm::ArrayRef<uint64_t> values) {
    auto& context  = module_->getContext();
    auto* int64_ty = llvm::Type::getInt64Ty(context);

    std::vector<llvm::Constant*> constants;
    constants.reserve(values.size());
    for (uint64_t val : values) {
      constants.push_back(llvm::ConstantInt::get(int64_ty, val));
    }

    auto* array_ty         = llvm::ArrayType::get(int64_ty, values.size());
    auto* constant_array   = llvm::ConstantArray::get(array_ty, constants);
    auto* gv               = create_global(name, array_ty, constant_array);
    auto constant_zero_i32 = llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), 0);
    return llvm::ConstantExpr::getGetElementPtr(  //
        array_ty,                                 //
        gv,                                       //
        llvm::ArrayRef<llvm::Value*>{constant_zero_i32, constant_zero_i32});
  }

  llvm::GlobalVariable* registerGlobalStruct(const std::string& name, int type_id, uint64_t type_size,
                                             uint64_t member_count, llvm::Constant* offset_ptr,
                                             llvm::Constant* members_data_ptr, llvm::Constant* count_ptr) {
    llvm::GlobalVariable* global_struct = create_global(name, struct_layout_type_);
    llvm::Constant* name_str            = create_global_constant_string(name);

    std::vector<llvm::Constant*> members = {
        ir_build.getInt32(type_id),
        name_str,
        ir_build.getInt64(type_size),
        ir_build.getInt64(member_count),
        offset_ptr,
        members_data_ptr,
        count_ptr,
        ir_build.getInt32(static_cast<int>(StructTypeFlag::USER_DEFINED))};  // TODO: use real type

    llvm::Constant* init = llvm::ConstantStruct::get(struct_layout_type_, members);

    global_struct->setInitializer(init);

    global_types_.global_type_data.try_emplace(
        name, GlobalTypeData::TypeData{init, global_struct, name_str, offset_ptr, count_ptr});

    return global_struct;
  }

  llvm::GlobalVariable* registerGlobalStructDecl(const std::string& name) {
    auto* global_struct = create_global(name, struct_layout_type_, nullptr, llvm::GlobalValue::ExternalWeakLinkage);
    // new llvm::GlobalVariable(*module_, struct_layout_type_, true, llvm::GlobalValue::ExternalWeakLinkage, nullptr,
    //                          helper::create_prefixed_name(name));
    // llvm::Constant* name_str = create_global_constant_string(name);
    global_types_.global_type_data.try_emplace(
        name, GlobalTypeData::TypeData{nullptr, global_struct, nullptr, nullptr, nullptr});

    return global_struct;
  }

  llvm::GlobalVariable* registerBuiltin(int type_id) {
    //  struct StructTypeInfo {
    //   int type_id;
    //   std::string name;
    //   size_t extent;
    //   size_t num_members;
    //   std::vector<size_t> offsets;
    //   std::vector<int> member_types;
    //   std::vector<size_t> array_sizes;
    //   StructTypeFlag flag;
    // };
    StructTypeInfo type_struct{type_id, type_db_->getTypeName(type_id), type_db_->getTypeSize(type_id), 1, {0}, {},
                               {1},     StructTypeFlag::USER_DEFINED};
    return registerTypeStruct(&type_struct);
  }

  llvm::GlobalVariable* registerTypeStruct(const StructTypeInfo* type_struct) {
    const auto name      = type_struct->name;
    const auto type_size = type_struct->extent;

    if (type_struct->flag == StructTypeFlag::FWD_DECL) {
      return registerGlobalStructDecl(name);
    }

    llvm::Constant* offset_ptr = create_global_array_ptr(helper::concat("offsets_", name), type_struct->offsets);
    llvm::Constant* count_ptr  = create_global_array_ptr(helper::concat("counts_", name), type_struct->array_sizes);

    llvm::Constant* members_array;
    llvm::Type* ptr_type{nullptr};  // TODO: make this unqual?
    std::vector<llvm::Constant*> member_types{};

    for (auto member_type_id : type_struct->member_types) {
      llvm::Constant* member = getOrRegister(member_type_id);
      if (ptr_type == nullptr) {
        ptr_type = member->getType();
      }
      member_types.emplace_back(member);
    }

    const auto member_count = type_struct->member_types.size();
    if (ptr_type != nullptr) {
      assert(member_count == type_struct->num_members);
      llvm::ArrayType* member_array_ty = llvm::ArrayType::get(ptr_type, member_count);
      llvm::Constant* init             = llvm::ConstantArray::get(member_array_ty, member_types);
      members_array                    = create_global(helper::concat("member_types_", name), member_array_ty, init);
    } else {
      llvm::Constant* null_member = llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(module_->getContext()));
      members_array               = null_member;
    }

    return registerGlobalStruct(name, type_struct->type_id, type_size, member_count, offset_ptr, members_array,
                                count_ptr);
  }

  llvm::GlobalVariable* registerUserDefined(int type_id) {
    const auto* const type_struct = type_db_->getStructInfo(type_id);
    return registerTypeStruct(type_struct);
  }

 public:
  llvm::Constant* getOrRegister(int type_id) {
    const auto name = type_db_->getTypeName(type_id);
    return module_->getOrInsertGlobal(helper::create_prefixed_name(name), struct_layout_type_,
                                      [&]() -> llvm::GlobalVariable* {
                                        LOG_DEBUG("Registering << " << type_id << " " << name)
                                        const bool is_builtin = type_db_->isBuiltinType(type_id);
                                        if (is_builtin) {
                                          return registerBuiltin(type_id);
                                        }
                                        return registerUserDefined(type_id);
                                      });
  }
};
}  // namespace typedb

class TypeRegistryGlobals final : public TypeRegistry {
  llvm::Module* module_;
  typedb::GlobalTypeRegistrar registrar_;

 public:
  TypeRegistryGlobals(llvm::Module& m, const TypeDatabase* type_db) : module_(&m), registrar_(&m, type_db) {
  }

  void registerModule(const ModuleData& m) override {
    for (const auto& type : m.types_list) {
      const auto type_id = registrar_.getOrRegister(type.type_id);
      LOG_DEBUG("Registering type_id " << type_id)
    }
  }

  llvm::Value* getOrRegister(llvm::Value* type_id_const) override {
    return registrar_.getOrRegister(helper::get_type_id(type_id_const));
  }
};

std::unique_ptr<TypeRegistry> get_type_id_handler(llvm::Module& m, const TypeDatabase* type_db,
                                                  const config::Configuration& configuration) {
  if (configuration[config::ConfigStdArgs::instrumentation]) {
    return std::make_unique<TypeRegistryGlobals>(m, type_db);
  }
  return std::make_unique<TypeRegistryNoOp>();
}

}  // namespace typeart
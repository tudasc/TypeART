#include "TypeIDProvider.h"

#include "TypeDB.h"
#include "TypeDatabase.h"
#include "TypeInterface.h"
#include "configuration/Configuration.h"
#include "instrumentation/TypeARTFunctions.h"
#include "support/ConfigurationBase.h"
#include "support/Logger.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>
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

void replace_whitespace_with_underscore(std::string& s) {
  std::replace_if(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c); }, '_');
}

inline int get_type_id(llvm::Value* type_id_const) {
  auto* constant_int = llvm::dyn_cast<llvm::ConstantInt>(type_id_const);
  assert(constant_int && "Expected llvm::ConstantInt");
  if (constant_int == nullptr) {
    return TYPEART_UNKNOWN_TYPE;
  }
  assert(constant_int->getBitWidth() <= 32 && "Type ID is too wide");
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
  std::string name = concat(llvm::StringRef{"_typeart_"}, std::forward<Args>(args)...);
  replace_whitespace_with_underscore(name);
  return name;
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
#if LLVM_VERSION_MAJOR > 17
    return global_type_data.contains(name);
#else
    return global_type_data.find(name) != global_type_data.end();
#endif
  }

  inline const TypeData& get_type(llvm::StringRef name) const {
#if LLVM_VERSION_MAJOR > 17
    return global_type_data.at(name);
#else
    return global_type_data.find(name)->second;
#endif
  }
};

struct GlobalTypeCallback {
  llvm::Module* module_;
  const TAFunctionQuery* f_query_;
  llvm::StringRef ctor_function_name{"__typeart_init_module_type_globals"};

 private:
  bool has_function() const {
    auto func = module_->getFunction(ctor_function_name);
    return func != nullptr;
  }

  llvm::BasicBlock* make_type_callback() const {
    using namespace llvm;
    const auto makeCtorFuncBody = [&]() -> BasicBlock* {
      auto& c                = module_->getContext();
      FunctionType* ctorType = FunctionType::get(llvm::Type::getVoidTy(c), false);
      Function* ctorFunction = Function::Create(ctorType, Function::PrivateLinkage, ctor_function_name, module_);
      BasicBlock* entry      = BasicBlock::Create(c, "entry", ctorFunction);
      auto* ret_inst         = ReturnInst::Create(c);
#if LLVM_VERSION_MAJOR > 17
      ret_inst->insertInto(entry, entry->getFirstInsertionPt());
#else
      entry->getInstList().push_back(ret_inst);
#endif

      llvm::appendToGlobalCtors(*module_, ctorFunction, 0, nullptr);

      return entry;
    };

    auto* func = module_->getFunction(ctor_function_name);
    if (func == nullptr) {
      return makeCtorFuncBody();
    }
    return &func->getEntryBlock();
  }

  llvm::BasicBlock* get_entry() {
    auto* func = module_->getFunction(ctor_function_name);
    if (func == nullptr) {
      return make_type_callback();
    }
    return &func->getEntryBlock();
  }

 public:
  GlobalTypeCallback(llvm::Module* module, const TAFunctionQuery* f_query) : module_(module), f_query_(f_query) {
  }

  void insert(llvm::Constant* global) {
    auto* block = get_entry();
    for (auto& inst : *block) {
      if (auto* call_base = llvm::dyn_cast<llvm::CallBase>(&inst)) {
        auto* argument = call_base->getArgOperand(0);
        if (global == argument) {
          LOG_DEBUG("Skipping, already contained");
          return;
        }
      }
    }
    llvm::IRBuilder<> IRB{&*block->getFirstInsertionPt()};

    IRB.CreateCall(f_query_->getFunctionFor(IFunc::type), llvm::ArrayRef<llvm::Value*>{global});
  }
};

struct GlobalTypeRegistrar {
  llvm::Module* module_;
  const TypeDatabase* type_db_;
  llvm::IRBuilder<> ir_build;
  GlobalTypeCallback type_callback;
  llvm::StructType* struct_layout_type_;
  GlobalTypeData global_types_;

  GlobalTypeRegistrar(llvm::Module* m, const TypeDatabase* type_db, const TAFunctionQuery* f_query)
      : module_(m), type_db_(type_db), ir_build(m->getContext()), type_callback(module_, f_query) {
    declare_layout();
  }

 private:
  void declare_layout() {
    auto& context = module_->getContext();
    llvm::IRBuilder<> Builder(context);
    struct_layout_type_ = llvm::StructType::create(context, "struct._typeart_struct_layout_t");
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
      llvm::GlobalVariable::LinkageTypes link_type = llvm::GlobalValue::WeakODRLinkage) const {
    // TODO: https://llvm.org/docs/LangRef.html#linkage w.r.t. forward declared types
    auto* global_struct =
        new llvm::GlobalVariable(*module_, type, true, link_type, init, helper::create_prefixed_name(name));
    return global_struct;
  }

  llvm::Constant* make_gep(llvm::Type* type, llvm::GlobalVariable* global) {
    auto* i32_zero_const = llvm::ConstantInt::get(ir_build.getInt32Ty(), 0);
    return llvm::ConstantExpr::getInBoundsGetElementPtr(
        type, global, llvm::ArrayRef<llvm::Constant*>{i32_zero_const, i32_zero_const});
  }

  llvm::Constant* create_global_constant_string(llvm::StringRef name) {
    // TODO think about linkage
    // auto* name_str = ir_build.CreateGlobalStringPtr(name, helper::create_prefixed_name("typename_", name), 0,
    // module_);
    auto* global_string =
        ir_build.CreateGlobalString(name, helper::create_prefixed_name("typename_", name), 0, module_);
    global_string->setConstant(true);
    global_string->setLinkage(llvm::GlobalValue::WeakODRLinkage);
    return make_gep(global_string->getValueType(), global_string);
  }

  llvm::Constant* create_global_array_ptr(const llvm::StringRef name, llvm::ArrayRef<uint64_t> values) {
    auto* int64_ty = ir_build.getInt64Ty();

    std::vector<llvm::Constant*> constants;
    constants.reserve(values.size());
    for (uint64_t val : values) {
      constants.push_back(llvm::ConstantInt::get(int64_ty, val));
    }

    auto* array_ty       = llvm::ArrayType::get(int64_ty, values.size());
    auto* constant_array = llvm::ConstantArray::get(array_ty, constants);
    auto* gv             = create_global(name, array_ty, constant_array);
    return make_gep(array_ty, gv);
  }

  llvm::GlobalVariable* registerGlobalStruct(const std::string& name, int type_id, uint64_t type_size,
                                             uint64_t member_count, llvm::Constant* offset_ptr,
                                             llvm::Constant* members_data_ptr, llvm::Constant* count_ptr,
                                             StructTypeFlag flag = StructTypeFlag::USER_DEFINED) {
    const auto name_struct              = flag == StructTypeFlag::FWD_DECL ? helper::concat(name, "_fwd") : name;
    llvm::GlobalVariable* global_struct = create_global(name_struct, struct_layout_type_);
    llvm::Constant* name_str            = create_global_constant_string(name);

    std::vector<llvm::Constant*> members = {ir_build.getInt32(type_id),
                                            name_str,
                                            ir_build.getInt64(type_size),
                                            ir_build.getInt64(member_count),
                                            offset_ptr,
                                            members_data_ptr,
                                            count_ptr,
                                            ir_build.getInt32(static_cast<int>(flag))};  // TODO: use real type

    llvm::Constant* init = llvm::ConstantStruct::get(struct_layout_type_, members);

    global_struct->setInitializer(init);

    global_types_.global_type_data.try_emplace(
        name, GlobalTypeData::TypeData{init, global_struct, name_str, offset_ptr, count_ptr});

    return global_struct;
  }

  llvm::GlobalVariable* registerTypeStruct(const StructTypeInfo* type_struct) {
    const auto name      = type_struct->name;
    const auto type_size = type_struct->extent;

    if (type_struct->flag == StructTypeFlag::FWD_DECL) {
      LOG_DEBUG("Type is forward decl " << name)
      // return registerGlobalStructDecl(name);
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
                                count_ptr, type_struct->flag);
  }

  llvm::GlobalVariable* registerBuiltin(int type_id) {
    auto type_name = type_db_->getTypeName(type_id);
    helper::replace_whitespace_with_underscore(type_name);
    StructTypeInfo type_struct{type_id, type_name, type_db_->getTypeSize(type_id), 1, {0},
                               {},      {1},       StructTypeFlag::BUILTIN};
    return registerTypeStruct(&type_struct);
  }

  llvm::GlobalVariable* registerUserDefined(int type_id) {
    const auto* const type_struct = type_db_->getStructInfo(type_id);
    if (type_struct == nullptr) {
      LOG_WARNING("Struct info is nullptr for id " << type_id)
    }
    return registerTypeStruct(type_struct);
  }

 public:
  llvm::Constant* getOrRegister(int type_id) {
    const auto name = type_db_->getTypeName(type_id);
    LOG_DEBUG(name << " aka " << helper::create_prefixed_name(name))
    return module_->getOrInsertGlobal(
        helper::create_prefixed_name(name), struct_layout_type_, [&]() -> llvm::GlobalVariable* {
          LOG_DEBUG("Registering << " << type_id << " " << name << " aka " << helper::create_prefixed_name(name))
          const bool is_builtin = type_db_->isBuiltinType(type_id);
          if (is_builtin) {
            auto* global = registerBuiltin(type_id);
            type_callback.insert(global);
            return global;
          }
          auto* global        = registerUserDefined(type_id);
          const auto fwd_decl = StructTypeFlag::FWD_DECL == type_db_->getStructInfo(type_id)->flag;
          if (!fwd_decl) {
            LOG_DEBUG("Registering forward declared variable " << *global)
          }
          type_callback.insert(global);
          return global;
        });
  }
};
}  // namespace typedb

class TypeRegistryGlobals final : public TypeRegistry {
  // llvm::Module* module_;
  typedb::GlobalTypeRegistrar registrar_;

 public:
  TypeRegistryGlobals(llvm::Module& m, const TypeDatabase* type_db, const TAFunctionQuery* f_query)
      : registrar_(&m, type_db, f_query) {
  }

  void registerModule(const ModuleData& m) override {
    for (const auto& type : m.types_list) {
      if (builtins::BuiltInQuery::is_builtin_type(type.type_id)) {
        continue;
      }
      if (!registrar_.type_db_->isValid(type.type_id)) {
        continue;
      }
      LOG_DEBUG("Registering type_id " << type.type_id)
      /*const auto* type_id =*/registrar_.getOrRegister(type.type_id);
    }
  }

  llvm::Value* getOrRegister(llvm::Value* type_id_const) override {
    return registrar_.getOrRegister(helper::get_type_id(type_id_const));
  }
};

class TypeRegistryAlternatives final : public TypeRegistry {
  TypeRegistryGlobals globals;
  TypeRegistryNoOp noops;

 public:
  TypeRegistryAlternatives(llvm::Module& m, const TypeDatabase* type_db, const TAFunctionQuery* f_query)
      : globals(m, type_db, f_query) {
  }

  void registerModule(const ModuleData& m) override {
    globals.registerModule(m);
  }

  llvm::Value* getOrRegister(llvm::Value* type_id_const) override {
    if (builtins::BuiltInQuery::is_builtin_type(helper::get_type_id(type_id_const))) {
      return noops.getOrRegister(type_id_const);
    }
    return globals.getOrRegister(type_id_const);
  }
};

std::unique_ptr<TypeRegistry> get_type_id_handler(llvm::Module& m, const TypeDatabase* type_db,
                                                  const config::Configuration& configuration,
                                                  const TAFunctionQuery* f_query) {
  TypeSerializationImplementation impl = configuration[config::ConfigStdArgs::type_serialization];
#if LLVM_VERSION_MAJOR < 15
  if (impl != typeart::TypeSerializationImplementation::FILE) {
    LOG_WARNING("Warning unsupported type serialization mode.")
  }
  // using llvm-14 would require opaque pointer mode for globals
  return std::make_unique<TypeRegistryNoOp>();
#else
  switch (impl) {
    case typeart::TypeSerializationImplementation::FILE:
      return std::make_unique<TypeRegistryNoOp>();
    case typeart::TypeSerializationImplementation::HYBRID:
      return std::make_unique<TypeRegistryAlternatives>(m, type_db, f_query);
    default:
      return std::make_unique<TypeRegistryGlobals>(m, type_db, f_query);
  }
#endif
}

}  // namespace typeart
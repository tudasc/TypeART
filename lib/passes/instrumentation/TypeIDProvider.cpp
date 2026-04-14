#include "TypeIDProvider.h"

#include "TypeDB.h"
#include "TypeDatabase.h"
#include "TypeInterface.h"
#include "configuration/Configuration.h"
#include "instrumentation/TypeARTFunctions.h"
#include "support/ConfigurationBase.h"
#include "support/Logger.h"

#include <cstdint>
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
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

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
  const auto str = (std::string{} + ... + std::string{std::forward<Args>(args)});
  return str;
}

template <typename... Args>
inline std::string create_prefixed_name(Args&&... args) {
  std::string name = concat("_typeart_", std::forward<Args>(args)...);
  replace_whitespace_with_underscore(name);
  return name;
}

inline bool is_forward_declaration(int type_id, const TypeDatabase& db) {
  if (db.isBuiltinType(type_id)) {
    return false;
  }
  const auto* struct_info = db.getStructInfo(type_id);
  return struct_info != nullptr && struct_info->flag == StructTypeFlag::FWD_DECL;
}

inline std::string get_link_name(int type_id, const TypeDatabase& db) {
  const auto base_name = db.getTypeName(type_id);
  return is_forward_declaration(type_id, db) ? concat(base_name, "_fwd") : base_name;
}

inline std::string get_prefixed_name(int type_id, const TypeDatabase& db) {
  return create_prefixed_name(get_link_name(type_id, db));
}

namespace detail {
template <typename T, typename SourceT>
T safe_cast(SourceT val) {
  // Check if value exceeds the maximum limit of the target type T
  // We cast max() to size_t to ensure we are comparing compatible types safely
  assert(static_cast<size_t>(val) <= static_cast<size_t>(std::numeric_limits<T>::max()) &&
         "Data loss detected: Value exceeds target type limits!");
  return static_cast<T>(val);
}
}  // namespace detail

template <typename T>
std::vector<T> get_serialized_members_for(const StructTypeInfo& info) {
  using namespace detail;
  std::vector<T> dest;
  const size_t required_space = info.offsets.size() + info.array_sizes.size() + 2;
  dest.reserve(required_space);

  // Layout : [ num_member, flag, offsets...[num_member], array_sizes...[num_member] ]

  dest.push_back(safe_cast<T>(info.num_members));
  dest.push_back(safe_cast<T>(static_cast<std::underlying_type_t<StructTypeFlag>>(info.flag)));

  for (size_t offset : info.offsets) {
    dest.push_back(safe_cast<T>(offset));
  }

  for (size_t size : info.array_sizes) {
    dest.push_back(safe_cast<T>(size));
  }

  return dest;
}

}  // namespace helper

namespace typedb {

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

enum class IGlobalType : short {
  type_id,
  name,
  extent,
  num_members,
  member_offsets,
  member_types,
  member_count,
  type_flag,
  ptr,
  info_holder
};

struct TypeHelper {
  llvm::IRBuilder<>& ir_build_;
  explicit TypeHelper(llvm::IRBuilder<>& ir_build) : ir_build_(ir_build) {
  }

  llvm::Type* get_type_for(IGlobalType type, bool as_array = false) {
    switch (type) {
      case IGlobalType::type_id:
      case IGlobalType::extent:
        return ir_build_.getInt32Ty();
      case IGlobalType::type_flag:
      case IGlobalType::num_members:
        return ir_build_.getInt16Ty();
      case IGlobalType::member_offsets:
      case IGlobalType::member_types:
      case IGlobalType::member_count: {
        if (as_array) {
          return ir_build_.getInt16Ty();
        }
#if LLVM_VERSION_MAJOR < 15
        return ir_build_.getInt8PtrTy();
#else
        return ir_build_.getPtrTy();
#endif
      }
      case IGlobalType::name:
      case IGlobalType::ptr:
      case IGlobalType::info_holder:
#if LLVM_VERSION_MAJOR < 15
        return ir_build_.getInt8PtrTy();
#else
        return ir_build_.getPtrTy();
#endif
    }
    llvm_unreachable("Should not be reached");
  }

  llvm::Constant* get_constant_for(IGlobalType type, size_t value) {
    switch (type) {
      case IGlobalType::type_id:
      case IGlobalType::extent:
        return ir_build_.getInt32(value);
      case IGlobalType::type_flag:
      case IGlobalType::num_members:
      case IGlobalType::member_offsets:
      case IGlobalType::member_count:
        return ir_build_.getInt16(value);
      default:
        break;
    }
    return ir_build_.getInt32(value);
  }

  llvm::Constant* get_constant_nullptr() {
    return llvm::ConstantPointerNull::get(llvm::dyn_cast<llvm::PointerType>(get_type_for(IGlobalType::ptr)));
  }
};

struct GlobalTypeRegistrar {
 private:
  llvm::Module* module_;
  const TypeDatabase* type_db_;
  llvm::IRBuilder<> ir_build;
  GlobalTypeCallback type_callback;
  llvm::StructType* struct_layout_type_;
  llvm::StructType* struct_layout_type_cold_;
  TypeHelper types_helper;
  const bool builtin_emit_name{false};

  void declare_layout() {
    auto& context       = module_->getContext();
    struct_layout_type_ = llvm::StructType::create(context, "struct._typeart_struct_layout_t");
    struct_layout_type_->setBody({
        types_helper.get_type_for(IGlobalType::type_id),  // uint32 type_id
        types_helper.get_type_for(IGlobalType::extent),   // uint32 extent
        types_helper.get_type_for(IGlobalType::info_holder),
    });
    struct_layout_type_cold_ = llvm::StructType::create(context, "struct._typeart_struct_layout_info_t");
    struct_layout_type_cold_->setBody({
        types_helper.get_type_for(IGlobalType::name),            // const char* name
        types_helper.get_type_for(IGlobalType::member_offsets),  // const uint16* offsets
        types_helper.get_type_for(IGlobalType::member_types),    // const typeart_struct_layout_t** member_types
    });
  }

  llvm::GlobalVariable* create_global(
      llvm::StringRef name, llvm::Type* type, llvm::Constant* init = nullptr,
      llvm::GlobalVariable::LinkageTypes link_type = llvm::GlobalValue::PrivateLinkage) const {
    auto* global_struct =
        new llvm::GlobalVariable(*module_, type, true, link_type, init, helper::create_prefixed_name(name));
    return global_struct;
  }

  llvm::Constant* create_global_constant_string(llvm::StringRef name, llvm::StringRef payload) {
    auto* global_string =
        ir_build.CreateGlobalString(payload, helper::create_prefixed_name("typename_", name), 0, module_);
    global_string->setConstant(true);
    global_string->setLinkage(llvm::GlobalValue::PrivateLinkage);
    return global_string;
  }

  template <typename InputRange, typename ConversionFunc>
  llvm::Constant* create_global_array_from_range(llvm::StringRef global_name, const InputRange& inputs,
                                                 llvm::Type* element_type, ConversionFunc&& convert_element) {
    if (inputs.empty()) {
      LOG_DEBUG("No values for global array, returning nullptr");
      return types_helper.get_constant_nullptr();
    }

    std::vector<llvm::Constant*> constants;
    constants.reserve(inputs.size());

    for (const auto& val : inputs) {
      constants.push_back(convert_element(val));
    }

    auto* array_ty       = llvm::ArrayType::get(element_type, inputs.size());
    auto* constant_array = llvm::ConstantArray::get(array_ty, constants);

    return create_global(global_name, array_ty, constant_array);
  }

  template <typename T>
  llvm::Constant* create_global_array_ptr(const llvm::StringRef name, llvm::ArrayRef<T> values,
                                          IGlobalType type = IGlobalType::member_offsets) {
    return create_global_array_from_range(name, values, types_helper.get_type_for(IGlobalType::member_offsets, true),
                                          [&](const T& val) { return types_helper.get_constant_for(type, val); });
  }

  llvm::Constant* create_global_member_array_ptr(const llvm::StringRef name, llvm::ArrayRef<int> member_types) {
    return create_global_array_from_range(name, member_types, types_helper.get_type_for(IGlobalType::member_types),
                                          [&](int member_id) { return getOrRegister(member_id); });
  }

  llvm::GlobalVariable* registerTypeStruct(const StructTypeInfo* type_struct) {
    const auto base_name = type_struct->name;
    const auto type_id   = type_struct->type_id;
    const bool is_fwd    = helper::is_forward_declaration(type_id, *type_db_);
    const auto link_name = helper::get_link_name(type_id, *type_db_);

    if (is_fwd) {
      LOG_DEBUG("Type is forward decl " << base_name)
    }

    const bool is_builtin = type_struct->flag == StructTypeFlag::BUILTIN;
    const bool emit_name  = !is_builtin || builtin_emit_name;

    llvm::GlobalVariable* global_struct =
        create_global(link_name, struct_layout_type_, nullptr, llvm::GlobalValue::LinkOnceODRLinkage);
    global_struct->setConstant(false);

    llvm::Comdat* comdat = this->module_->getOrInsertComdat(helper::create_prefixed_name(link_name));
    comdat->setSelectionKind(llvm::Comdat::Any);
    global_struct->setComdat(comdat);

    auto add_to_comdat = [&](llvm::Constant* ptr) {
      if (auto* global = llvm::dyn_cast_or_null<llvm::GlobalObject>(ptr)) {
        global->setComdat(comdat);
      }
    };

    const auto get_info_object = [&]() -> llvm::Constant* {
      if (emit_name) {
        llvm::Constant* name_str_ptr = create_global_constant_string(link_name, base_name);
        const auto info_data         = helper::get_serialized_members_for<uint16_t>(*type_struct);
        llvm::Constant* data_ptr =
            create_global_array_ptr<uint16_t>(helper::concat("info_data_", link_name), info_data);
        llvm::Constant* members_ptr =
            create_global_member_array_ptr(helper::concat("member_types_", link_name), type_struct->member_types);

        llvm::GlobalVariable* global_struct_info =
            create_global(helper::concat(link_name, "_info"), struct_layout_type_cold_, nullptr);

        std::vector<llvm::Constant*> init_fields_cold = {name_str_ptr, data_ptr, members_ptr};
        global_struct_info->setInitializer(llvm::ConstantStruct::get(struct_layout_type_cold_, init_fields_cold));

        {
          add_to_comdat(global_struct_info);
          add_to_comdat(data_ptr);
          add_to_comdat(members_ptr);
          add_to_comdat(name_str_ptr);
        }

        return global_struct_info;
      }
      return types_helper.get_constant_nullptr();
    };

    std::vector<llvm::Constant*> init_fields = {
        types_helper.get_constant_for(IGlobalType::type_id, type_struct->type_id),
        types_helper.get_constant_for(IGlobalType::extent, type_struct->extent), get_info_object()};
    llvm::Constant* init = llvm::ConstantStruct::get(struct_layout_type_, init_fields);
    global_struct->setInitializer(init);

    return global_struct;
  }

  llvm::GlobalVariable* registerBuiltin(int type_id) {
    auto type_name = type_db_->getTypeName(type_id);
    helper::replace_whitespace_with_underscore(type_name);
    StructTypeInfo type_struct{type_id, type_name, type_db_->getTypeSize(type_id), 1, {},
                               {},      {},        StructTypeFlag::BUILTIN};
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
  GlobalTypeRegistrar(llvm::Module* m, const TypeDatabase* type_db, const TAFunctionQuery* f_query)
      : module_(m),
        type_db_(type_db),
        ir_build(m->getContext()),
        type_callback(module_, f_query),
        types_helper(ir_build) {
    declare_layout();
  }

  const TypeDatabase& db() const {
    return *type_db_;
  }

  llvm::Constant* getOrRegister(int type_id) {
    const auto base_name     = type_db_->getTypeName(type_id);
    const auto prefixed_name = helper::get_prefixed_name(type_id, *type_db_);

    LOG_DEBUG(base_name << " aka " << prefixed_name)

    return module_->getOrInsertGlobal(prefixed_name, struct_layout_type_, [&]() -> llvm::GlobalVariable* {
      LOG_DEBUG("Registering << " << type_id << " " << base_name << " aka " << prefixed_name)

      if (type_db_->isBuiltinType(type_id)) {
        auto* global = registerBuiltin(type_id);
        type_callback.insert(global);
        return global;
      }

      auto* global = registerUserDefined(type_id);

      if (!helper::is_forward_declaration(type_id, *type_db_)) {
        LOG_DEBUG("Registering forward declared variable " << *global)
      }
      type_callback.insert(global);
      return global;
    });
  }
};  // namespace typedb
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
      if (!registrar_.db().isValid(type.type_id)) {
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
    LOG_WARNING("Unsupported type serialization mode for LLVM-" << LLVM_VERSION_MAJOR)
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
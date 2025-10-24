#include "TypeIDProvider.h"

#include "TypeDatabase.h"
#include "configuration/Configuration.h"
#include "support/ConfigurationBase.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Value.h>
#include <memory>

namespace typeart {

class TypeRegistryNoOp final : public TypeRegistry {
 public:
  llvm::Value* getOrRegister(llvm::Value* type_id_const) override {
    return type_id_const;
  }
};

namespace helper {
llvm::Constant* create_global_array_ptr(llvm::Module& M, llvm::LLVMContext& C, llvm::ArrayRef<uint64_t> values,
                                        const llvm::Twine& name) {
  auto* int64_ty = llvm::Type::getInt64Ty(C);
  std::vector<llvm::Constant*> constants;
  constants.reserve(values.size());
  for (uint64_t val : values) {
    constants.push_back(llvm::ConstantInt::get(int64_ty, val));
  }

  auto* array_ty         = llvm::ArrayType::get(int64_ty, values.size());
  auto* constant_array   = llvm::ConstantArray::get(array_ty, constants);
  auto* gv               = new llvm::GlobalVariable(M, array_ty,                         //
                                                    true,                                //
                                                    llvm::GlobalValue::InternalLinkage,  //
                                                    constant_array,                      //
                                                    name);
  auto constant_zero_i32 = llvm::ConstantInt::get(llvm::Type::getInt32Ty(C), 0);
  return llvm::ConstantExpr::getGetElementPtr(  //
      array_ty,                                 //
      gv,                                       //
      llvm::ArrayRef<llvm::Value*>{constant_zero_i32, constant_zero_i32});
}
}  // namespace helper

namespace typedb {
struct GlobalTypeRegistrar {
  llvm::Module* module_;
  llvm::StructType* struct_layout_type_;
  const TypeDatabase* type_db_;

  GlobalTypeRegistrar() = default;
  GlobalTypeRegistrar(llvm::Module* m, llvm::StructType* struct_layout_type, const TypeDatabase* type_db)
      : module_(m), struct_layout_type_(struct_layout_type), type_db_(type_db) {
  }

  llvm::GlobalVariable* getOrRegister(int type_id) {
    llvm::GlobalVariable* global_struct_test =
        new llvm::GlobalVariable(*module_, struct_layout_type_, false, llvm::GlobalValue::InternalLinkage,
                                 nullptr,  // init later
                                 "type_id_test"

        );
    auto& context = module_->getContext();
    llvm::IRBuilder<> Builder(context);
    llvm::Constant* NameStr              = Builder.CreateGlobalStringPtr("sample struct", "str.name", 0, module_);
    llvm::Constant* OffsetsPtr           = helper::create_global_array_ptr(*module_, context, {0, 8}, "offsets.arr");
    llvm::Constant* CountPtr             = helper::create_global_array_ptr(*module_, context, {1, 1}, "count.arr");
    std::vector<llvm::Constant*> Members = {Builder.getInt32(0),   //
                                            NameStr,               //
                                            Builder.getInt64(24),  //
                                            Builder.getInt64(2),   //
                                            OffsetsPtr,            //
                                            global_struct_test,    //
                                            CountPtr};
    llvm::Constant* TheInitializer       = llvm::ConstantStruct::get(struct_layout_type_, Members);

    global_struct_test->setInitializer(TheInitializer);
    return global_struct_test;
  }
};
}  // namespace typedb

class TypeRegistryGlobals final : public TypeRegistry {
  llvm::Module* module_;
  llvm::StructType* struct_layout_type_;
  typedb::GlobalTypeRegistrar registrar_;

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
        llvm::PointerType::getUnqual(context)   // const size_t* count
    });
  }

 public:
  TypeRegistryGlobals(llvm::Module& m, const TypeDatabase* type_db) : module_(&m) {
    declareLayout();
    registrar_ = typedb::GlobalTypeRegistrar(module_, struct_layout_type_, type_db);
  }

  llvm::Value* getOrRegister(llvm::Value* type_id_const) override {
    auto* constant_int = llvm::dyn_cast<llvm::ConstantInt>(type_id_const);
    const int type_id  = static_cast<int>(constant_int->getSExtValue());
    return registrar_.getOrRegister(type_id);
  }
};

std::unique_ptr<TypeRegistry> get_type_id_handler(llvm::Module& m, const TypeDatabase* type_gen,
                                                  const config::Configuration& configuration) {
  if (configuration[config::ConfigStdArgs::instrumentation]) {
    return std::make_unique<TypeRegistryGlobals>(m, type_gen);
  }
  return std::make_unique<TypeRegistryNoOp>();
}

}  // namespace typeart
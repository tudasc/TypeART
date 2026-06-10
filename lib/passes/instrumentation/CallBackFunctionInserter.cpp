#include "CallBackFunctionInserter.h"

#include "configuration/Configuration.h"
#include "instrumentation/TypeIDProvider.h"
#include "support/ConfigurationBase.h"
#include "support/Logger.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"

#include <memory>

namespace typeart {

class CallbackFunctionInserter final : public InstrumentationInserter {
  std::unique_ptr<TypeRegistry> type_id_handler_;
  TAFunctionQuery* function_query_;
  TypeSerializationImplementation mode_;
  // bool mixed_mode{false};

 private:
  llvm::CallInst* create_instrumentation_call(llvm::IRBuilder<>& IRB, IFunc callback_type,
                                              llvm::Value* instruction_or_value, InstrumentationPayload args);

 public:
  CallbackFunctionInserter(const config::Configuration& configuration, std::unique_ptr<TypeRegistry> type_id_handler,
                           TAFunctionQuery* function_query);

  llvm::CallInst* insert_heap_instrumentation(llvm::IRBuilder<>& IRB, llvm::CallBase* heap_call,
                                              InstrumentationPayload) override;

  llvm::CallInst* insert_stack_instrumentation(llvm::IRBuilder<>& IRB, llvm::Instruction* alloca,
                                               InstrumentationPayload) override;

  llvm::CallInst* insert_global_instrumentation(llvm::IRBuilder<>& IRB, llvm::GlobalValue* global_var,
                                                InstrumentationPayload) override;

  llvm::CallInst* insert_free_instrumentation(llvm::IRBuilder<>& IRB, llvm::CallBase* call,
                                              llvm::Value* pointer_value) override;
};

CallbackFunctionInserter::CallbackFunctionInserter(const config::Configuration& configuration,
                                                   std::unique_ptr<TypeRegistry> type_id_handler,
                                                   TAFunctionQuery* function_query)
    : type_id_handler_(std::move(type_id_handler)), function_query_(function_query) {
  mode_ = configuration[config::ConfigStdArgs::type_serialization];
  // mixed_mode                            = value == TypeSerializationImplementation::HYBRID;
}

// Private Helper Definition
llvm::CallInst* CallbackFunctionInserter::create_instrumentation_call(llvm::IRBuilder<>& IRB, IFunc callback_type,
                                                                      llvm::Value* instruction_or_value,
                                                                      InstrumentationPayload args) {
  const auto callback_id = ifunc_for_function(callback_type, instruction_or_value);
  auto type_id_param_out = type_id_handler_->getOrRegister(args.typeid_value);

  const bool has_global_type_payload = !llvm::isa<llvm::ConstantInt>(type_id_param_out);
  const auto mode                    = has_global_type_payload ? mode_ : TypeSerializationImplementation::FILE;
  auto function                      = function_query_->getFunctionFor(callback_id, mode);

  return IRB.CreateCall(function,
                        llvm::ArrayRef<llvm::Value*>{args.pointer_value, type_id_param_out, args.element_count});
}

llvm::CallInst* CallbackFunctionInserter::insert_heap_instrumentation(llvm::IRBuilder<>& IRB, llvm::CallBase* heap_call,
                                                                      InstrumentationPayload args) {
  return create_instrumentation_call(IRB, IFunc::heap, heap_call, args);
}

llvm::CallInst* CallbackFunctionInserter::insert_stack_instrumentation(llvm::IRBuilder<>& IRB,
                                                                       llvm::Instruction* alloca,
                                                                       InstrumentationPayload args) {
  return create_instrumentation_call(IRB, IFunc::stack, alloca, args);
}

llvm::CallInst* CallbackFunctionInserter::insert_global_instrumentation(llvm::IRBuilder<>& IRB,
                                                                        llvm::GlobalValue* global_var,
                                                                        InstrumentationPayload args) {
  return create_instrumentation_call(IRB, IFunc::global, global_var, args);
}

llvm::CallInst* CallbackFunctionInserter::insert_free_instrumentation(llvm::IRBuilder<>& IRB, llvm::CallBase* call,
                                                                      llvm::Value* pointer_value) {
  const auto callback_id = ifunc_for_function(IFunc::free, call);
  return IRB.CreateCall(function_query_->getFunctionFor(callback_id), llvm::ArrayRef<llvm::Value*>{pointer_value});
}

std::unique_ptr<InstrumentationInserter> make_callback_inserter(const config::Configuration& configuration,
                                                                std::unique_ptr<TypeRegistry> type_id_handler,
                                                                TAFunctionQuery* function_query) {
  return std::make_unique<CallbackFunctionInserter>(configuration, std::move(type_id_handler), function_query);
}

}  // namespace typeart

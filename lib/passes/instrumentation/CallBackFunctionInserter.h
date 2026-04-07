#ifndef TYPEART_CALLBACKFUNCTIONINSERTER_H
#define TYPEART_CALLBACKFUNCTIONINSERTER_H

#include "instrumentation/TypeARTFunctions.h"

#include "llvm/IR/IRBuilder.h"

#include <memory>

namespace llvm {
class CallInst;
class Value;
class CallBase;
class Instruction;
class GlobalValue;
}  // namespace llvm

namespace typeart {
namespace config {
class Configuration;
}
class TypeRegistry;
class TAFunctionQuery;

struct InstrumentationPayload {
  llvm::Value* pointer_value;
  llvm::Value* element_count;
  llvm::Value* typeid_value;
};

class InstrumentationInserter {
 public:
  virtual ~InstrumentationInserter() = default;

  virtual llvm::CallInst* insert_heap_instrumentation(llvm::IRBuilder<>& IRB, llvm::CallBase* heap_call,
                                                      InstrumentationPayload) = 0;

  virtual llvm::CallInst* insert_stack_instrumentation(llvm::IRBuilder<>& IRB, llvm::Instruction* alloca,
                                                       InstrumentationPayload) = 0;

  virtual llvm::CallInst* insert_global_instrumentation(llvm::IRBuilder<>& IRB, llvm::GlobalValue* global_var,
                                                        InstrumentationPayload) = 0;

  virtual llvm::CallInst* insert_free_instrumentation(llvm::IRBuilder<>& IRB, llvm::CallBase* heap_call,
                                                      llvm::Value* pointer_value) = 0;
};

std::unique_ptr<InstrumentationInserter> make_callback_inserter(const config::Configuration& configuration,
                                                                std::unique_ptr<TypeRegistry> type_id_handler,
                                                                TAFunctionQuery* function_query);

}  // namespace typeart

#endif  // TYPEART_CALLBACKFUNCTIONINSERTER_H

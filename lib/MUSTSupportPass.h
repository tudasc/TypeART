#ifndef LIB_MUST_SUPPORT_PASS_H
#define LIB_MUST_SUPPORT_PASS_H

#include "llvm/Pass.h"

namespace llvm {
class Constant;
}

namespace must {
namespace pass {

class MustSupportPass : public llvm::BasicBlockPass {
 public:
  static char ID;  // used to identify pass

  /* Call base class ctor with ID */
  MustSupportPass() : llvm::BasicBlockPass(ID) {}
  /* Run once per module */
  bool doInitialization(llvm::Module& m) override;
  /* Run once per function */
  bool doInitialization(llvm::Function& f) override;
  /* Runs on every basic block */
  bool runOnBasicBlock(llvm::BasicBlock& BB) override;
  /* Run once per module */
  bool doFinalization(llvm::Module& m) override;

 private:
  bool isAllocateFunction(std::string identifier) { return allocFunctions.find(identifier) != allocFunctions.end(); }

  bool isDeallocateFunction(std::string identifier) {
    return deallocFunctions.find(identifier) != deallocFunctions.end();
  }

  void setFunctionLinkageExternal(llvm::Constant* c);
  /*
   * Declares the external functions in the module.
   * void __must_support_alloc(void *ptr_base, int type_id, long int count, long int elem_size)
  * void __must_support_free(void *ptr)
  */
  void declareInstrumentationFunctions(llvm::Module& m);
  void propagateTypeInformation(llvm::Module& m);

  /** Data members */
  std::string allocInstrumentation{"__must_support_alloc"};
  std::string freeInstrumentation{"__must_support_free"};

  /** Look up sets for keyword strings */
  const std::set<std::string> allocFunctions{"malloc"};
  const std::set<std::string> deallocFunctions{"free"};
};

}  // pass
}  // must

#endif

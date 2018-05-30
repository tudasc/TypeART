#ifndef _LIB_MUSTSUPPORTPASS_H
#define _LIB_MUSTSUPPORTPASS_H

#include "TypeDB.h"
#include "TypeManager.h"
#include "llvm/Pass.h"

#include <set>

namespace llvm {
class Constant;
class raw_ostream;
}  // namespace llvm

namespace must {
namespace pass {

class MustSupportPass : public llvm::BasicBlockPass {
 public:
  static char ID;  // used to identify pass

  /* Call base class ctor with ID */
  MustSupportPass();

  /* Run once per module */
  bool doInitialization(llvm::Module& m) override;
  /* Run once per function */
  bool doInitialization(llvm::Function& f) override;
  /* Runs on every basic block */
  bool runOnBasicBlock(llvm::BasicBlock& BB) override;
  /* Run once per module */
  bool doFinalization(llvm::Module& m) override;

 private:
  void setFunctionLinkageExternal(llvm::Constant* c);
  /*
   * Declares the external functions in the module.
   * void __must_support_alloc(void *ptr_base, int type_id, long int count, long int elem_size)
  * void __must_support_free(void *ptr)
  */
  void declareInstrumentationFunctions(llvm::Module& m);
  void propagateTypeInformation(llvm::Module& m);

  // std::string type2String(llvm::Type* type);
  // int retrieveTypeID(llvm::Type* type);

  void printStats(llvm::raw_ostream&);

  /** Data members */
  std::string allocInstrumentation{"__must_support_alloc"};
  std::string freeInstrumentation{"__must_support_free"};

  std::string configFileName{"musttypes"};

  // static std::unique_ptr<TypeMapping> typeMapping;

  TypeManager typeManager;

  std::string configFile;
};

}  // namespace pass
}  // namespace must

#endif

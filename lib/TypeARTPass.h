#ifndef _LIB_MUSTSUPPORTPASS_H
#define _LIB_MUSTSUPPORTPASS_H

#include "TypeDB.h"
#include "TypeManager.h"
#include "llvm/Pass.h"

#include "string"

namespace llvm {
class Constant;
class Module;
class Function;
class AnalysisUsage;
}  // namespace llvm

namespace typeart {
namespace pass {

class TypeArtSupportPass : public llvm::FunctionPass {
 private:
  struct TypeArtFunc {
    const std::string name{""};
    llvm::Constant* f{nullptr};
  };

  TypeArtFunc typeart_alloc{"__typeart_alloc"};

  TypeArtFunc typeart_free{"__typeart_free"};

  std::string configFileName{"types.yaml"};

  TypeManager typeManager;

  std::string configFile;

 public:
  static char ID;  // used to identify pass

  /* Call base class ctor with ID */
  TypeArtSupportPass();

  /* Run once per module */
  bool doInitialization(llvm::Module&) override;
  /* Runs on every basic block */
  bool runOnFunction(llvm::Function&) override;
  /* Run once per module */
  bool doFinalization(llvm::Module&) override;

  void getAnalysisUsage(llvm::AnalysisUsage&) const override;

 private:
  /*
   * Declares the external functions in the module.
   * void __typeart_alloc(void *ptr_base, int type_id, long int count, long int elem_size)
   * void __typeart_free(void *ptr)
   */
  void declareInstrumentationFunctions(llvm::Module&);
  void propagateTypeInformation(llvm::Module&);
  void printStats(llvm::raw_ostream&);
};

}  // namespace pass
}  // namespace typeart

#endif

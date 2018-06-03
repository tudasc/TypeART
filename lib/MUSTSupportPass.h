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

class MustSupportPass : public llvm::FunctionPass {
 private:
  struct TypeArtFunc {
    const std::string name{""};
    llvm::Constant* f{nullptr};
  };

  TypeArtFunc typeart_alloc{"__must_support_alloc"};

  TypeArtFunc typeart_free{"__must_support_free"};

  std::string configFileName{"musttypes"};

  TypeManager typeManager;

  std::string configFile;

 public:
  static char ID;  // used to identify pass

  /* Call base class ctor with ID */
  MustSupportPass();

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
   * void __must_support_alloc(void *ptr_base, int type_id, long int count, long int elem_size)
   * void __must_support_free(void *ptr)
   */
  void declareInstrumentationFunctions(llvm::Module& m);
  void propagateTypeInformation(llvm::Module& m);

  // std::string type2String(llvm::Type* type);
  // int retrieveTypeID(llvm::Type* type);

  void printStats(llvm::raw_ostream&);
};

}  // namespace pass
}  // namespace must

#endif

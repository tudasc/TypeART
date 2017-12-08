#ifndef LIB_MUST_SUPPORT_PASS_H
#define LIB_MUST_SUPPORT_PASS_H

#include "llvm/Pass.h"

namespace must {
	namespace pass {

		class MustSupportPass : public llvm::BasicBlockPass {
			/* Run once per module */
			bool doInitialization(llvm::Module &m) override;
			/* Run once per function */
			bool doInitialization(llvm::Function &f) override;
			/* Runs on every basic block */
			bool runOnBasicBlock(llvm::BasicBlock &BB) override;
			/* Run once per module */
			bool doFinalization(llvm::Module &m) override;
		};

	} // pass
} // must

#endif

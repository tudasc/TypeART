#include "MUSTSupportPass.h"

using namespace llvm;

namespace must {
	namespace pass {

		bool MustSupportPass::doInitialization(Module &m) {}

		bool MustSupportPass::doInitialization(Function &f) {}

		bool MustSupportPass::runOnBasicBlock(BasicBlock &bb) {}

		bool MustSupportPass::doFinalization(Module &m) {}

	} // pass
} // must

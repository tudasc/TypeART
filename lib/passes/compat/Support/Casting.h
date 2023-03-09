#ifndef TYPEART_LIB_PASSES_COMPAT_SUPPORT_CASTING_H
#define TYPEART_LIB_PASSES_COMPAT_SUPPORT_CASTING_H

#if LLVM_VERSION_MAJOR < 11
#include <llvm/Support/Compiler.h>

namespace llvm {
template <typename First, typename Second, typename... Rest, typename Y>
LLVM_NODISCARD inline bool isa(const Y& Val) {
  return isa<First>(Val) || isa<Second, Rest...>(Val);
}
}  // namespace llvm
#endif

#endif  // TYPEART_LIB_PASSES_COMPAT_SUPPORT_CASTING_H

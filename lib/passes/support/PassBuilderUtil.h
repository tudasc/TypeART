
// Taken and adapted from llvm/Passes/PassBuilder.h
#ifndef D0C8104D_3805_4D3E_AEDA_E6B36227C166
#define D0C8104D_3805_4D3E_AEDA_E6B36227C166

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace typeart::util::pass {

inline bool checkParametrizedPassName(llvm::StringRef Name, llvm::StringRef PassName) {
  using namespace llvm;
  if (!Name.consume_front(PassName))
    return false;
  // normal pass name w/o parameters == default parameters
  if (Name.empty())
    return true;
#if LLVM_VERSION_MAJOR > 14
  return Name.starts_with("<") && Name.ends_with(">");
#else
  return Name.startswith("<") && Name.endswith(">");
#endif
}

/// This performs customized parsing of pass name with parameters.
///
/// We do not need parametrization of passes in textual pipeline very often,
/// yet on a rare occasion ability to specify parameters right there can be
/// useful.
///
/// \p Name - parameterized specification of a pass from a textual pipeline
/// is a string in a form of :
///      PassName '<' parameter-list '>'
///
/// Parameter list is being parsed by the parser callable argument, \p Parser,
/// It takes a string-ref of parameters and returns either StringError or a
/// parameter list in a form of a custom parameters type, all wrapped into
/// Expected<> template class.
///
template <typename ParametersParseCallableT>
inline auto parsePassParameters(ParametersParseCallableT&& Parser, llvm::StringRef Name, llvm::StringRef PassName)
    -> decltype(Parser(llvm::StringRef{})) {
  using namespace llvm;
  using ParametersT = typename decltype(Parser(StringRef{}))::value_type;

  StringRef Params = Name;
  if (!Params.consume_front(PassName)) {
    llvm_unreachable("unable to strip pass name from parametrized pass specification");
  }
  if (!Params.empty() && (!Params.consume_front("<") || !Params.consume_back(">"))) {
    llvm_unreachable("invalid format for parametrized pass name");
  }

  Expected<ParametersT> Result = Parser(Params);
  assert((Result || Result.template errorIsA<StringError>()) && "Pass parameter parser can only return StringErrors.");
  return Result;
}

}  // namespace typeart::util::pass

#endif /* D0C8104D_3805_4D3E_AEDA_E6B36227C166 */

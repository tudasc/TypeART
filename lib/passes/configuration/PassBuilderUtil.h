
// TypeART library
//
// Copyright (c) 2017-2025 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

// Taken and adapted from llvm/Passes/PassBuilder.h
#ifndef TYPEART_PASS_BUILDER_UTIL_H
#define TYPEART_PASS_BUILDER_UTIL_H

#include "support/Logger.h"

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
#if LLVM_VERSION_MAJOR > 15
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

#endif /* TYPEART_PASS_BUILDER_UTIL_H */

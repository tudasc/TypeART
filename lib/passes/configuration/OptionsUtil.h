#ifndef C09614E8_ACD7_462B_9534_B29749F6EAD7
#define C09614E8_ACD7_462B_9534_B29749F6EAD7

#include "analysis/MemInstFinder.h"
#include "support/Logger.h"
#include "typegen/TypeGenerator.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

namespace typeart::config::util {

template <typename... Strings>
bool with_any_of(llvm::StringRef lhs, Strings&&... rhs) {
  return !lhs.empty() && ((lhs == rhs) || ...);
}

template <typename ClType>
ClType string_to_enum(llvm::StringRef cl_value) {
  using ::typeart::TypegenImplementation;
  using ::typeart::analysis::FilterImplementation;
  if constexpr (std::is_same_v<TypegenImplementation, ClType>) {
    auto val = llvm::StringSwitch<ClType>(cl_value)
                   .Case("ir", TypegenImplementation::IR)
                   .Case("dimeta", TypegenImplementation::DIMETA)
                   .Default(TypegenImplementation::DIMETA);
    return val;
  } else {
    auto val = llvm::StringSwitch<ClType>(cl_value)
                   .Case("cg", FilterImplementation::cg)
                   .Case("none", FilterImplementation::none)
                   .Case("std", FilterImplementation::standard)
                   .Default(FilterImplementation::standard);
    return val;
  }
}

template <typename ClType>
ClType make_opt(llvm::StringRef cl_value) {
  LOG_DEBUG("Parsing value " << cl_value)
  if constexpr (std::is_same_v<bool, ClType>) {
    const bool is_true_val  = with_any_of(cl_value, "true", "TRUE", "1");
    const bool is_false_val = with_any_of(cl_value, "false", "FALSE", "0");
    if (!(is_true_val || is_false_val)) {
      LOG_WARNING("Illegal bool value")
    }
    assert((is_true_val || is_false_val) && "Illegal bool value for environment flag");
    return is_true_val;
  } else {
    if constexpr (std::is_enum_v<ClType>) {
      auto enum_value = string_to_enum<ClType>(cl_value);
      return enum_value;
    } else {
      return std::string{cl_value};
    }
  }
}

}  // namespace typeart::config::util
#endif /* C09614E8_ACD7_462B_9534_B29749F6EAD7 */

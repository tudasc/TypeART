

#include "PassConfiguration.h"

#include "OptionsUtil.h"
#include "analysis/MemInstFinder.h"
#include "support/ConfigurationBase.h"
#include "support/ConfigurationBaseOptions.h"
#include "support/Error.h"

namespace typeart::config::pass {

namespace detail {

template <typename ClType>
ClType make_entry(llvm::StringRef key) {
  return ClType{};
}

}  // namespace detail

struct PassStdArgsEq final {
#define TYPEART_CONFIG_OPTION(name, path, type, def_value, description, upper_path) \
  static constexpr char name[] = path "=";
#include "support/ConfigurationBaseOptions.h"
#undef TYPEART_CONFIG_OPTION
};

llvm::Expected<TypeARTConfigOptions> parse_typeart_config(llvm::StringRef parameters) {
  TypeARTConfigOptions result;
  bool global_set{false};
  while (!parameters.empty()) {
    llvm::StringRef parameter_name;
    std::tie(parameter_name, parameters) = parameters.split(';');

    const bool enable = !parameter_name.consume_front("no-");

    if (parameter_name == ConfigStdArgs::heap) {
      result.heap = enable;
      continue;
    }

    if (parameter_name == ConfigStdArgs::stack) {
      result.stack = enable;
      continue;
    }

    if (parameter_name == ConfigStdArgs::global) {
      global_set    = true;
      result.global = enable;
      continue;
    }

    if (parameter_name == ConfigStdArgs::filter) {
      result.filter = enable;
      continue;
    }

    if (parameter_name == ConfigStdArgs::stats) {
      result.statistics = enable;
      continue;
    }

    if (parameter_name == ConfigStdArgs::stack_lifetime) {
      result.stack_lifetime = enable;
      continue;
    }

    if (parameter_name.consume_front(PassStdArgsEq::typegen)) {
      result.typegen = util::string_to_enum<TypegenImplementation>(parameter_name);
      continue;
    }

    if (parameter_name.consume_front(PassStdArgsEq::filter_impl)) {
      result.filter_config.implementation = util::string_to_enum<analysis::FilterImplementation>(parameter_name);
      continue;
    }
  }
  if (!global_set) {
    // Stack implies global
    result.global = result.stack;
  }
  return result;
}

}  // namespace typeart::config::pass
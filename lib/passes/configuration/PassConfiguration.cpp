

#include "PassConfiguration.h"

#include "OptionsUtil.h"
#include "analysis/MemInstFinder.h"
#include "configuration/Configuration.h"
#include "support/ConfigurationBase.h"
#include "support/ConfigurationBaseOptions.h"
#include "support/Error.h"
#include "support/Logger.h"

#include "llvm/Support/FormatVariadicDetails.h"

#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>
#include <string>

namespace typeart::config::pass {

struct PassStdArgsEq final {
#define TYPEART_CONFIG_OPTION(name, path, type, def_value, description, upper_path) \
  static constexpr char name[] = path "=";
#include "support/ConfigurationBaseOptions.h"
#undef TYPEART_CONFIG_OPTION
};

llvm::Expected<TypeARTConfigOptions> parse_typeart_config(llvm::StringRef parameters) {
  return parse_typeart_config_with_occurrence(parameters).first;
}

PassConfig parse_typeart_config_with_occurrence(llvm::StringRef parameters) {
  LOG_DEBUG("Parsing string: " << parameters)
  TypeARTConfigOptions result;
  OptOccurrenceMap occurrence_map;
  bool global_set{false};
  bool stack_set{false};

  while (!parameters.empty()) {
    llvm::StringRef parameter_name;
    std::tie(parameter_name, parameters) = parameters.split(';');

    const bool enable = !parameter_name.consume_front("no-");

    if (parameter_name == ConfigStdArgs::heap) {
      result.heap                         = enable;
      occurrence_map[ConfigStdArgs::heap] = true;
      continue;
    }

    if (parameter_name == ConfigStdArgs::stack) {
      result.stack                         = enable;
      occurrence_map[ConfigStdArgs::stack] = true;
      stack_set                            = true;
      continue;
    }

    if (parameter_name == ConfigStdArgs::global) {
      global_set                            = true;
      result.global                         = enable;
      occurrence_map[ConfigStdArgs::global] = true;
      continue;
    }

    if (parameter_name == ConfigStdArgs::filter) {
      result.filter                         = enable;
      occurrence_map[ConfigStdArgs::filter] = true;
      continue;
    }

    if (parameter_name == ConfigStdArgs::stats) {
      result.statistics                    = enable;
      occurrence_map[ConfigStdArgs::stats] = true;
      continue;
    }

    if (parameter_name == ConfigStdArgs::stack_lifetime) {
      result.stack_lifetime                         = enable;
      occurrence_map[ConfigStdArgs::stack_lifetime] = true;
      continue;
    }

    if (parameter_name.consume_front(PassStdArgsEq::typegen)) {
      result.typegen                         = util::string_to_enum<TypegenImplementation>(parameter_name);
      occurrence_map[ConfigStdArgs::typegen] = true;
      continue;
    }

    if (parameter_name.consume_front(PassStdArgsEq::filter_impl)) {
      result.filter_config.implementation        = util::string_to_enum<analysis::FilterImplementation>(parameter_name);
      occurrence_map[ConfigStdArgs::filter_impl] = true;
      continue;
    }

    if (parameter_name.consume_front(PassStdArgsEq::filter_glob)) {
      result.filter_config.glob                  = parameter_name;
      occurrence_map[ConfigStdArgs::filter_glob] = true;
      continue;
    }
    if (parameter_name.consume_front(PassStdArgsEq::filter_glob_deep)) {
      result.filter_config.glob_deep                  = parameter_name;
      occurrence_map[ConfigStdArgs::filter_glob_deep] = true;
      continue;
    }
    if (parameter_name == ConfigStdArgs::analysis_filter_global) {
      result.analysis_config.filter_global                  = enable;
      occurrence_map[ConfigStdArgs::analysis_filter_global] = true;
      continue;
    }

    if (parameter_name == ConfigStdArgs::analysis_filter_alloca_non_array) {
      result.analysis_config.filter_alloca_non_array                  = enable;
      occurrence_map[ConfigStdArgs::analysis_filter_alloca_non_array] = true;
      continue;
    }

    if (parameter_name == ConfigStdArgs::analysis_filter_heap_alloc) {
      result.analysis_config.filter_heap_alloc                  = enable;
      occurrence_map[ConfigStdArgs::analysis_filter_heap_alloc] = true;
      continue;
    }

    if (parameter_name == ConfigStdArgs::analysis_filter_pointer_alloc) {
      result.analysis_config.filter_pointer_alloc                  = enable;
      occurrence_map[ConfigStdArgs::analysis_filter_pointer_alloc] = true;
      continue;
    }
    {
      // undefined symbol issue: llvm::formatv("Unknown TypeART option {0} with unparsed list {1}", parameter_name,
      // parameters).str()
      std::string out_string;
      llvm::raw_string_ostream out_stream(out_string);
      out_stream << "Unknown TypeART option " << parameter_name;
      if (!parameters.empty()) {
        out_stream << " with unparsed list " << parameters;
      }
      return {error::make_string_error(out_stream.str()), occurrence_map};
    }
  }
  if (!global_set && stack_set) {
    // Stack implies global
    result.global                         = result.stack;
    occurrence_map[ConfigStdArgs::global] = true;
  }
  return {result, occurrence_map};
}

}  // namespace typeart::config::pass
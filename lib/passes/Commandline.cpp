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

#include "Commandline.h"

#include "analysis/MemInstFinder.h"
#include "support/Configuration.h"
#include "support/Logger.h"
#include "typegen/TypeGenerator.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/CommandLine.h"

#include <charconv>
#include <string>
#include <string_view>
#include <type_traits>

using namespace llvm;

namespace typeart::config {

namespace util {
std::optional<std::string> get_env_flag(std::string_view flag) {
  const char* env_value = std::getenv(flag.data());
  const bool exists     = env_value != nullptr;
  if (exists) {
    LOG_DEBUG("Using env var " << flag << "=" << env_value)
    return std::string{env_value};
  }
  LOG_DEBUG("Not using env var " << flag << "=<unset>")
  return {};
}

std::string get_type_file_path() {
  auto flag_value = get_env_flag("TYPEART_TYPE_FILE");
  return flag_value.value_or("types.yaml");
}

}  // namespace util

namespace cl {
struct CommandlineStdArgs final {
#define TYPEART_CONFIG_OPTION(name, path, type, def_value, description, upper_path) \
  static constexpr char name[] = "typeart-" path;
#include "support/ConfigurationBaseOptions.h"
#undef TYPEART_CONFIG_OPTION
};
}  // namespace cl

}  // namespace typeart::config

using typeart::config::ConfigStdArgDescriptions;
using typeart::config::ConfigStdArgTypes;
using typeart::config::ConfigStdArgValues;
using typeart::config::cl::CommandlineStdArgs;

cl::OptionCategory typeart_category("TypeART instrumentation pass", "These control the instrumentation.");

static cl::opt<ConfigStdArgTypes::types_ty> cl_typeart_type_file(CommandlineStdArgs::types,
                                                                 cl::desc(ConfigStdArgDescriptions::types),
                                                                 cl::cat(typeart_category));

static cl::opt<ConfigStdArgTypes::stats_ty> cl_typeart_stats(CommandlineStdArgs::stats,
                                                             cl::desc(ConfigStdArgDescriptions::stats), cl::Hidden,
                                                             cl::init(ConfigStdArgValues::stats),
                                                             cl::cat(typeart_category));

static cl::opt<ConfigStdArgTypes::heap_ty> cl_typeart_instrument_heap(CommandlineStdArgs::heap,
                                                                      cl::desc(ConfigStdArgDescriptions::heap),
                                                                      cl::init(ConfigStdArgValues::heap),
                                                                      cl::cat(typeart_category));

static cl::opt<ConfigStdArgTypes::global_ty> cl_typeart_instrument_global(CommandlineStdArgs::global,
                                                                          cl::desc(ConfigStdArgDescriptions::global),
                                                                          cl::init(ConfigStdArgValues::global),
                                                                          cl::cat(typeart_category));

static cl::opt<ConfigStdArgTypes::stack_ty> cl_typeart_instrument_stack(CommandlineStdArgs::stack,
                                                                        cl::desc(ConfigStdArgDescriptions::stack),
                                                                        cl::init(ConfigStdArgValues::stack),
                                                                        cl::cat(typeart_category),
                                                                        cl::callback([](const bool& opt) {
                                                                          if (opt) {
                                                                            ::cl_typeart_instrument_global = true;
                                                                          }
                                                                        }));

static cl::opt<ConfigStdArgTypes::stack_lifetime_ty> cl_typeart_instrument_stack_lifetime(
    CommandlineStdArgs::stack_lifetime, cl::desc(ConfigStdArgDescriptions::stack_lifetime),
    cl::init(ConfigStdArgValues::stack_lifetime), cl::cat(typeart_category));

static cl::opt<typeart::TypegenImplementation> cl_typeart_typegen_implementation(
    CommandlineStdArgs::typegen, cl::desc(ConfigStdArgDescriptions::typegen),
    cl::values(clEnumValN(typeart::TypegenImplementation::IR, "ir", "Standard IR based type parser"),
               clEnumValN(typeart::TypegenImplementation::DIMETA, "dimeta", "Metadata-based parser (default)")),
    cl::Hidden, cl::init(typeart::TypegenImplementation::DIMETA), cl::cat(typeart_category));

static cl::OptionCategory typeart_analysis_category(
    "TypeART memory instruction finder", "These options control which memory instructions are collected/filtered.");

static cl::opt<ConfigStdArgTypes::analysis_filter_alloca_non_array_ty> cl_typeart_filter_stack_non_array(
    CommandlineStdArgs::analysis_filter_alloca_non_array,
    cl::desc(ConfigStdArgDescriptions::analysis_filter_alloca_non_array), cl::Hidden,
    cl::init(ConfigStdArgValues::analysis_filter_alloca_non_array), cl::cat(typeart_analysis_category));

static cl::opt<ConfigStdArgTypes::analysis_filter_heap_alloc_ty> cl_typeart_filter_heap_alloc(
    CommandlineStdArgs::analysis_filter_heap_alloc, cl::desc(ConfigStdArgDescriptions::analysis_filter_heap_alloc),
    cl::Hidden, cl::init(ConfigStdArgValues::analysis_filter_heap_alloc), cl::cat(typeart_analysis_category));

static cl::opt<ConfigStdArgTypes::analysis_filter_global_ty> cl_typeart_filter_global(
    CommandlineStdArgs::analysis_filter_global, cl::desc(ConfigStdArgDescriptions::analysis_filter_global), cl::Hidden,
    cl::init(ConfigStdArgValues::analysis_filter_global), cl::cat(typeart_analysis_category));

static cl::opt<ConfigStdArgTypes::filter_ty> cl_typeart_call_filter(CommandlineStdArgs::filter,
                                                                    cl::desc(ConfigStdArgDescriptions::filter),
                                                                    cl::Hidden, cl::init(ConfigStdArgValues::filter),
                                                                    cl::cat(typeart_analysis_category));

static cl::opt<typeart::analysis::FilterImplementation> cl_typeart_call_filter_implementation(
    CommandlineStdArgs::filter_impl, cl::desc(ConfigStdArgDescriptions::filter_impl),
    cl::values(clEnumValN(typeart::analysis::FilterImplementation::none, "none", "No filter"),
               clEnumValN(typeart::analysis::FilterImplementation::standard, "std",
                          "Standard forward filter (default)"),
               clEnumValN(typeart::analysis::FilterImplementation::cg, "cg", "Call-graph-based filter")),
    cl::Hidden, cl::init(typeart::analysis::FilterImplementation::standard), cl::cat(typeart_analysis_category));

static cl::opt<ConfigStdArgTypes::filter_glob_ty> cl_typeart_call_filter_glob(
    CommandlineStdArgs::filter_glob, cl::desc(ConfigStdArgDescriptions::filter_glob), cl::Hidden,
    cl::init(ConfigStdArgValues::filter_glob), cl::cat(typeart_analysis_category));

static cl::opt<ConfigStdArgTypes::filter_glob_deep_ty> cl_typeart_call_filter_glob_deep(
    CommandlineStdArgs::filter_glob_deep, cl::desc(ConfigStdArgDescriptions::filter_glob_deep), cl::Hidden,
    cl::init(ConfigStdArgValues::filter_glob_deep), cl::cat(typeart_analysis_category));

static cl::opt<ConfigStdArgTypes::filter_cg_file_ty> cl_typeart_call_filter_cg_file(
    CommandlineStdArgs::filter_cg_file, cl::desc(ConfigStdArgDescriptions::filter_cg_file), cl::Hidden,
    cl::init(ConfigStdArgValues::filter_cg_file), cl::cat(typeart_analysis_category));

static cl::opt<ConfigStdArgTypes::analysis_filter_pointer_alloc_ty> cl_typeart_filter_pointer_alloca(
    CommandlineStdArgs::analysis_filter_pointer_alloc,
    cl::desc(ConfigStdArgDescriptions::analysis_filter_pointer_alloc), cl::Hidden,
    cl::init(ConfigStdArgValues::analysis_filter_pointer_alloc), cl::cat(typeart_analysis_category));

namespace typeart::config::cl {

// std::string get_type_file_path() {
// if (!cl_typeart_type_file.empty()) {
//   LOG_DEBUG("Using cl::opt for types file " << cl_typeart_type_file.getValue());
//   return cl_typeart_type_file.getValue();
// }
//   const char* type_file = std::getenv("TYPEART_TYPE_FILE");
//   if (type_file != nullptr) {
//     LOG_DEBUG("Using env var for types file " << type_file)
//     return std::string{type_file};
//   }
//   LOG_DEBUG("Loading default types file types.yaml");
//   return "types.yaml";
// }

namespace detail {
template <typename ClOpt>
config::OptionValue make_opt(const ClOpt& cl_opt) {
  if constexpr (std::is_base_of_v<llvm::cl::Option, ClOpt>) {
    if constexpr (std::is_enum_v<decltype(cl_opt.getValue())>) {
      return config::OptionValue{static_cast<int>(cl_opt.getValue())};
    } else {
      return config::OptionValue{cl_opt.getValue()};
    }
  } else {
    return config::OptionValue{cl_opt};
  }
}

template <typename ClOpt>
std::pair<StringRef, typename CommandLineOptions::OptionsMap::mapped_type> make_entry(std::string&& key,
                                                                                      ClOpt&& cl_opt) {
  return {key, make_opt(std::forward<ClOpt>(cl_opt))};
}

template <typename ClOpt>
std::pair<StringRef, typename CommandLineOptions::ClOccurrenceMap::mapped_type> make_occurr_entry(std::string&& key,
                                                                                                  ClOpt&& cl_opt) {
  return {key, (cl_opt.getNumOccurrences() > 0)};
}
}  // namespace detail

CommandLineOptions::CommandLineOptions() {
  using namespace config;
  using namespace typeart::config::cl::detail;

  const auto type_file = [&]() {
    if (!cl_typeart_type_file.empty()) {
      LOG_DEBUG("Using cl::opt for types file " << cl_typeart_type_file.getValue());
      return cl_typeart_type_file.getValue();
    }
    return util::get_type_file_path();
  }();

  mapping_ = {
      make_entry(ConfigStdArgs::types, type_file),
      make_entry(ConfigStdArgs::stats, cl_typeart_stats),
      make_entry(ConfigStdArgs::heap, cl_typeart_instrument_heap),
      make_entry(ConfigStdArgs::global, cl_typeart_instrument_global),
      make_entry(ConfigStdArgs::stack, cl_typeart_instrument_stack),
      make_entry(ConfigStdArgs::stack_lifetime, cl_typeart_instrument_stack_lifetime),
      make_entry(ConfigStdArgs::typegen, cl_typeart_typegen_implementation),
      make_entry(ConfigStdArgs::filter, cl_typeart_call_filter),
      make_entry(ConfigStdArgs::filter_impl, cl_typeart_call_filter_implementation),
      make_entry(ConfigStdArgs::filter_glob, cl_typeart_call_filter_glob),
      make_entry(ConfigStdArgs::filter_glob_deep, cl_typeart_call_filter_glob_deep),
      make_entry(ConfigStdArgs::filter_cg_file, cl_typeart_call_filter_cg_file),
      make_entry(ConfigStdArgs::analysis_filter_global, cl_typeart_filter_global),
      make_entry(ConfigStdArgs::analysis_filter_heap_alloc, cl_typeart_filter_heap_alloc),
      make_entry(ConfigStdArgs::analysis_filter_pointer_alloc, cl_typeart_filter_pointer_alloca),
      make_entry(ConfigStdArgs::analysis_filter_alloca_non_array, cl_typeart_filter_stack_non_array),
  };

  occurence_mapping_ = {
      make_occurr_entry(ConfigStdArgs::types, cl_typeart_type_file),
      make_occurr_entry(ConfigStdArgs::stats, cl_typeart_stats),
      make_occurr_entry(ConfigStdArgs::heap, cl_typeart_instrument_heap),
      make_occurr_entry(ConfigStdArgs::global, cl_typeart_instrument_global),
      make_occurr_entry(ConfigStdArgs::stack, cl_typeart_instrument_stack),
      make_occurr_entry(ConfigStdArgs::stack_lifetime, cl_typeart_instrument_stack_lifetime),
      make_occurr_entry(ConfigStdArgs::typegen, cl_typeart_typegen_implementation),
      make_occurr_entry(ConfigStdArgs::filter, cl_typeart_call_filter),
      make_occurr_entry(ConfigStdArgs::filter_impl, cl_typeart_call_filter_implementation),
      make_occurr_entry(ConfigStdArgs::filter_glob, cl_typeart_call_filter_glob),
      make_occurr_entry(ConfigStdArgs::filter_glob_deep, cl_typeart_call_filter_glob_deep),
      make_occurr_entry(ConfigStdArgs::filter_cg_file, cl_typeart_call_filter_cg_file),
      make_occurr_entry(ConfigStdArgs::analysis_filter_global, cl_typeart_filter_global),
      make_occurr_entry(ConfigStdArgs::analysis_filter_heap_alloc, cl_typeart_filter_heap_alloc),
      make_occurr_entry(ConfigStdArgs::analysis_filter_pointer_alloc, cl_typeart_filter_pointer_alloca),
      make_occurr_entry(ConfigStdArgs::analysis_filter_alloca_non_array, cl_typeart_filter_stack_non_array),
  };
}

std::optional<typeart::config::OptionValue> CommandLineOptions::getValue(std::string_view opt_path) const {
  auto key = llvm::StringRef(opt_path.data());
  if (mapping_.count(key) != 0U) {
    return mapping_.lookup(key);
  }
  return {};
}

[[maybe_unused]] bool CommandLineOptions::valueSpecified(std::string_view opt_path) const {
  auto key = llvm::StringRef(opt_path.data());
  return occurence_mapping_.lookup(key);
}

}  // namespace typeart::config::cl

namespace typeart::config::env {
struct EnvironmentStdArgs final {
#define TYPEART_CONFIG_OPTION(name, path, type, def_value, description, upper_path) \
  static constexpr char name[] = "TYPEART_" upper_path;
#include "support/ConfigurationBaseOptions.h"
#undef TYPEART_CONFIG_OPTION
};

struct EnvironmentStdArgsValues final {
#define TYPEART_CONFIG_OPTION(name, path, type, def_value, description, upper_path) \
  static constexpr char name[] = #def_value;
#include "support/ConfigurationBaseOptions.h"
#undef TYPEART_CONFIG_OPTION
};

}  // namespace typeart::config::env

using typeart::config::ConfigStdArgDescriptions;
using typeart::config::ConfigStdArgTypes;
using typeart::config::ConfigStdArgValues;
using typeart::config::env::EnvironmentStdArgs;

namespace typeart::config::env {

namespace detail {
template <typename... Strings>
bool with_any_of(std::string_view lhs, Strings&&... rhs) {
  return !lhs.empty() && ((lhs == rhs) || ...);
}

template <typename ClType>
int enum_to_int(std::string_view cl_value) {
  if constexpr (std::is_same_v<typeart::TypegenImplementation, ClType>) {
    auto val = llvm::StringSwitch<ClType>(cl_value.data())
                   .Case("ir", typeart::TypegenImplementation::IR)
                   .Case("dimeta", typeart::TypegenImplementation::DIMETA)
                   .Default(typeart::TypegenImplementation::DIMETA);
    return static_cast<int>(val);
  } else {
    auto val = llvm::StringSwitch<ClType>(cl_value.data())
                   .Case("cg", typeart::analysis::FilterImplementation::cg)
                   .Case("none", typeart::analysis::FilterImplementation::none)
                   .Case("std", typeart::analysis::FilterImplementation::standard)
                   .Default(typeart::analysis::FilterImplementation::standard);
    return static_cast<int>(val);
  }
}

template <typename ClType>
config::OptionValue make_opt(std::string_view cl_value) {
  LOG_DEBUG("Parsing value " << cl_value)
  if constexpr (std::is_same_v<bool, ClType>) {
    const bool is_true_val  = with_any_of(cl_value, "true", "TRUE", "1");
    const bool is_false_val = with_any_of(cl_value, "false", "FALSE", "0");
    if (!(is_true_val || is_false_val)) {
      LOG_WARNING("Illegal bool value")
    }
    assert((is_true_val || is_false_val) && "Illegal bool value for environment flag");
    return config::OptionValue{is_true_val};
  } else {
    if constexpr (std::is_enum_v<ClType>) {
      return config::OptionValue{enum_to_int<ClType>(cl_value)};
    } else {
      return config::OptionValue{std::string{cl_value}};
    }
  }
}

template <typename ClType>
std::pair<StringRef, typename EnvironmentFlagsOptions::OptionsMap::mapped_type> make_entry(
    std::string&& key, std::string_view cl_opt, const std::string& default_value) {
  const auto env_value = util::get_env_flag(cl_opt);
  return {key, make_opt<ClType>(env_value.value_or(default_value))};
}

template <typename ClOpt>
std::pair<StringRef, typename EnvironmentFlagsOptions::ClOccurrenceMap::mapped_type> make_occurr_entry(
    std::string&& key, ClOpt&& cl_opt) {
  const bool occured = (util::get_env_flag(cl_opt).has_value());
  LOG_DEBUG("Key :" << key << ":" << occured)
  return {key, occured};
}
}  // namespace detail

EnvironmentFlagsOptions::EnvironmentFlagsOptions() {
  using namespace config;
  using namespace typeart::config::env::detail;

  LOG_DEBUG("Construct environment flag options")

  mapping_ = {
      make_entry<std::string>(ConfigStdArgs::types, "TYPEART_TYPE_FILE", ConfigStdArgValues::types),
      make_entry<ConfigStdArgTypes::stats_ty>(ConfigStdArgs::stats, EnvironmentStdArgs::stats,
                                              EnvironmentStdArgsValues::stats),
      make_entry<ConfigStdArgTypes::heap_ty>(ConfigStdArgs::heap, EnvironmentStdArgs::heap,
                                             EnvironmentStdArgsValues::heap),
      make_entry<ConfigStdArgTypes::global_ty>(ConfigStdArgs::global, EnvironmentStdArgs::global,
                                               EnvironmentStdArgsValues::global),
      make_entry<ConfigStdArgTypes::stack_ty>(ConfigStdArgs::stack, EnvironmentStdArgs::stack,
                                              EnvironmentStdArgsValues::stack),
      make_entry<ConfigStdArgTypes::stack_lifetime_ty>(
          ConfigStdArgs::stack_lifetime, EnvironmentStdArgs::stack_lifetime, EnvironmentStdArgsValues::stack_lifetime),
      make_entry<typeart::TypegenImplementation>(ConfigStdArgs::typegen, EnvironmentStdArgs::typegen,
                                                 ConfigStdArgValues::typegen),
      make_entry<ConfigStdArgTypes::filter_ty>(ConfigStdArgs::filter, EnvironmentStdArgs::filter,
                                               EnvironmentStdArgsValues::filter),
      make_entry<typeart::analysis::FilterImplementation>(ConfigStdArgs::filter_impl, EnvironmentStdArgs::filter_impl,
                                                          ConfigStdArgValues::filter_impl),

      make_entry<ConfigStdArgTypes::filter_glob_ty>(ConfigStdArgs::filter_glob, EnvironmentStdArgs::filter_glob,
                                                    ConfigStdArgValues::filter_glob),
      make_entry<ConfigStdArgTypes::filter_glob_deep_ty>(
          ConfigStdArgs::filter_glob_deep, EnvironmentStdArgs::filter_glob_deep, ConfigStdArgValues::filter_glob_deep),
      make_entry<ConfigStdArgTypes::filter_cg_file_ty>(
          ConfigStdArgs::filter_cg_file, EnvironmentStdArgs::filter_cg_file, ConfigStdArgValues::filter_cg_file),
      make_entry<ConfigStdArgTypes::analysis_filter_global_ty>(ConfigStdArgs::analysis_filter_global,
                                                               EnvironmentStdArgs::analysis_filter_global,
                                                               EnvironmentStdArgsValues::analysis_filter_global),
      make_entry<ConfigStdArgTypes::analysis_filter_heap_alloc_ty>(
          ConfigStdArgs::analysis_filter_heap_alloc, EnvironmentStdArgs::analysis_filter_heap_alloc,
          EnvironmentStdArgsValues::analysis_filter_heap_alloc),
      make_entry<ConfigStdArgTypes::analysis_filter_pointer_alloc_ty>(
          ConfigStdArgs::analysis_filter_pointer_alloc, EnvironmentStdArgs::analysis_filter_pointer_alloc,
          EnvironmentStdArgsValues::analysis_filter_pointer_alloc),
      make_entry<ConfigStdArgTypes::analysis_filter_alloca_non_array_ty>(
          ConfigStdArgs::analysis_filter_alloca_non_array, EnvironmentStdArgs::analysis_filter_alloca_non_array,
          EnvironmentStdArgsValues::analysis_filter_alloca_non_array),
  };

  occurence_mapping_ = {
      make_occurr_entry(ConfigStdArgs::types, "TYPEART_TYPE_FILE"),
      make_occurr_entry(ConfigStdArgs::stats, EnvironmentStdArgs::stats),
      make_occurr_entry(ConfigStdArgs::heap, EnvironmentStdArgs::heap),
      make_occurr_entry(ConfigStdArgs::global, EnvironmentStdArgs::global),
      make_occurr_entry(ConfigStdArgs::stack, EnvironmentStdArgs::stack),
      make_occurr_entry(ConfigStdArgs::stack_lifetime, EnvironmentStdArgs::stack_lifetime),
      make_occurr_entry(ConfigStdArgs::typegen, EnvironmentStdArgs::typegen),
      make_occurr_entry(ConfigStdArgs::filter, EnvironmentStdArgs::filter),
      make_occurr_entry(ConfigStdArgs::filter_impl, EnvironmentStdArgs::filter_impl),
      make_occurr_entry(ConfigStdArgs::filter_glob, EnvironmentStdArgs::filter_glob),
      make_occurr_entry(ConfigStdArgs::filter_glob_deep, EnvironmentStdArgs::filter_glob_deep),
      make_occurr_entry(ConfigStdArgs::filter_cg_file, EnvironmentStdArgs::filter_cg_file),
      make_occurr_entry(ConfigStdArgs::analysis_filter_global, EnvironmentStdArgs::analysis_filter_global),
      make_occurr_entry(ConfigStdArgs::analysis_filter_heap_alloc, EnvironmentStdArgs::analysis_filter_heap_alloc),
      make_occurr_entry(ConfigStdArgs::analysis_filter_pointer_alloc,
                        EnvironmentStdArgs::analysis_filter_pointer_alloc),
      make_occurr_entry(ConfigStdArgs::analysis_filter_alloca_non_array,
                        EnvironmentStdArgs::analysis_filter_alloca_non_array),
  };

  if (!occurence_mapping_.lookup(ConfigStdArgs::global)) {
    const auto stack_value          = mapping_.lookup(ConfigStdArgs::stack);
    mapping_[ConfigStdArgs::global] = OptionValue{static_cast<bool>(stack_value)};
  }
}

std::optional<typeart::config::OptionValue> EnvironmentFlagsOptions::getValue(std::string_view opt_path) const {
  auto key = llvm::StringRef(opt_path.data());
  if (mapping_.count(key) != 0U) {
    return mapping_.lookup(key);
  }
  return {};
}

[[maybe_unused]] bool EnvironmentFlagsOptions::valueSpecified(std::string_view opt_path) const {
  auto key = llvm::StringRef(opt_path.data());
  return occurence_mapping_.lookup(key);
}

}  // namespace typeart::config::env

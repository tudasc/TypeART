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

#include "EnvironmentConfiguration.h"

#include "Configuration.h"
#include "OptionsUtil.h"
#include "PassConfiguration.h"
#include "configuration/TypeARTOptions.h"
#include "support/ConfigurationBase.h"
#include "support/Logger.h"
#include "support/Util.h"

#include "llvm/ADT/StringSwitch.h"

#include <charconv>
#include <string>
#include <string_view>
#include <type_traits>

using namespace llvm;

namespace typeart::config::env {
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

}  // namespace typeart::config::env

namespace typeart::config::env {

struct EnvironmentStdArgsValues final {
#define TYPEART_CONFIG_OPTION(name, path, type, def_value, description, upper_path) \
  static constexpr char name[] = #def_value;
#include "support/ConfigurationBaseOptions.h"
#undef TYPEART_CONFIG_OPTION
};

namespace detail {
template <typename ClType>
OptionValue make_opt(std::string_view cl_value) {
  // LOG_DEBUG("Parsing value " << cl_value)
  auto value = util::make_opt<ClType>(cl_value.data());
  if constexpr (std::is_enum_v<ClType>) {
    return OptionValue{static_cast<int>(value)};
  } else {
    return OptionValue{value};
  }
}

template <typename ClType>
std::pair<StringRef, typename OptionsMap::mapped_type> make_entry(std::string_view key, std::string_view cl_opt,
                                                                  const std::string& default_value) {
  const auto env_value = get_env_flag(cl_opt);
  return {key, detail::make_opt<ClType>(env_value.value_or(default_value))};
}

template <typename ClOpt>
std::pair<StringRef, typename OptOccurrenceMap::mapped_type> make_occurr_entry(std::string_view key, ClOpt&& cl_opt) {
  const bool occurred = (get_env_flag(cl_opt).has_value());
  return {key, occurred};
}
}  // namespace detail

EnvironmentFlagsOptions::EnvironmentFlagsOptions() {
  using namespace config;
  using namespace typeart::config::env::detail;

  LOG_DEBUG("Construct environment flag options")

  mapping_ = {
      make_entry<std::string>(ConfigStdArgs::types, config::EnvironmentStdArgs::types, ConfigStdArgValues::types),
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
      make_occurr_entry(ConfigStdArgs::types, config::EnvironmentStdArgs::types),
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

  if (!occurence_mapping_.lookup(ConfigStdArgs::global) && occurence_mapping_.lookup(ConfigStdArgs::stack)) {
    const auto stack_value                    = mapping_.lookup(ConfigStdArgs::stack);
    mapping_[ConfigStdArgs::global]           = OptionValue{static_cast<bool>(stack_value)};
    occurence_mapping_[ConfigStdArgs::global] = true;
  }

  auto result = pass::parse_typeart_config_with_occurrence(get_env_flag("TYPEART_OPTIONS").value_or(""));
  if (!result.first) {
    LOG_INFO("No parseable TYPEART_OPTIONS: " << result.first.takeError())
  } else {
    LOG_DEBUG("Parsed TYPEART_OPTIONS\n" << *result.first)
    const auto typeart_options = helper::options_to_map(result.first.get());
    for (const auto& entry : result.second) {
      const auto key = entry.getKey();
      // LOG_DEBUG("Looking at " << key << " " << entry.second << ":" << occurence_mapping_[key])
      if (entry.second && !occurence_mapping_[key]) {  // single ENV priority over TYPEART_OPTIONS
        LOG_DEBUG("Replacing " << key)
        mapping_[key]           = typeart_options.lookup(key);
        occurence_mapping_[key] = true;
      }
    }
  }
}

std::optional<typeart::config::OptionValue> EnvironmentFlagsOptions::getValue(std::string_view opt_path) const {
  auto key = llvm::StringRef(opt_path.data());
  if (occurence_mapping_.lookup(key)) {  // only have value if it occurred
    return mapping_.lookup(key);
  }
  return {};
}

[[maybe_unused]] bool EnvironmentFlagsOptions::valueSpecified(std::string_view opt_path) const {
  auto key = llvm::StringRef(opt_path.data());
  return occurence_mapping_.lookup(key);
}

}  // namespace typeart::config::env

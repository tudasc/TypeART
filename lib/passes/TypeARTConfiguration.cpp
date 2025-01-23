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

#include "TypeARTConfiguration.h"

#include "Commandline.h"
#include "support/FileConfiguration.h"
#include "support/Logger.h"
#include <string>

namespace typeart::config {

TypeARTConfiguration::TypeARTConfiguration(std::unique_ptr<file::FileOptions> config_options,
                                           std::unique_ptr<cl::CommandLineOptions> commandline_options,
                                           std::unique_ptr<env::EnvironmentFlagsOptions> env_options)
    : configuration_options_(std::move(config_options)),
      commandline_options_(std::move(commandline_options)),
      env_options_(std::move(env_options)) {
}

std::optional<OptionValue> TypeARTConfiguration::getValue(std::string_view opt_path) const {
  auto get_value = [&](const auto& options, const char* source) -> std::optional<OptionValue> {
    LOG_DEBUG("Query " << source << " " << opt_path.data())
    if (prioritize_commandline && options.valueSpecified(opt_path)) {
      LOG_DEBUG("Take " << source << " arg for " << opt_path.data());
      return options.getValue(opt_path);
    }
    return std::nullopt;
  };
  if (auto value = get_value(*env_options_, "ENV")) {
    return value;
  }
  if (auto value = get_value(*commandline_options_, "CL")) {
    return value;
  }
  return configuration_options_->getValue(opt_path);
}

OptionValue TypeARTConfiguration::getValueOr(std::string_view opt_path, OptionValue alt) const {
  auto get_value = [&](const auto& options, const char* source) -> std::optional<OptionValue> {
    LOG_DEBUG("Query " << source << " " << opt_path.data())
    if (prioritize_commandline && options.valueSpecified(opt_path)) {
      LOG_DEBUG("Take " << source << " arg for " << opt_path.data());
      return options.getValueOr(opt_path, alt);
    }
    return std::nullopt;
  };
  if (auto value = get_value(*env_options_, "ENV")) {
    return value.value();
  }
  if (auto value = get_value(*commandline_options_, "CL")) {
    return value.value();
  }
  return configuration_options_->getValueOr(opt_path, alt);
}

OptionValue TypeARTConfiguration::operator[](std::string_view opt_path) const {
  auto get_value = [&](const auto& options, const char* source) -> std::optional<OptionValue> {
    LOG_DEBUG("Query " << source << " " << opt_path.data())
    if (prioritize_commandline && options.valueSpecified(opt_path)) {
      LOG_DEBUG("Take " << source << " arg for " << opt_path.data());
      return options.operator[](opt_path);
    }
    return std::nullopt;
  };
  if (auto value = get_value(*env_options_, "ENV")) {
    return value.value();
  }
  if (auto value = get_value(*commandline_options_, "CL")) {
    return value.value();
  }
  auto result = configuration_options_->operator[](opt_path);
  LOG_DEBUG("Query file " << static_cast<std::string>(result))
  return result;
}

void TypeARTConfiguration::prioritizeCommandline(bool do_prioritize) {
  prioritize_commandline = do_prioritize;
}

void TypeARTConfiguration::emitTypeartFileConfiguration(llvm::raw_ostream& out_stream) {
  out_stream << configuration_options_->getConfigurationAsString();
}

llvm::ErrorOr<std::unique_ptr<TypeARTConfiguration>> make_typeart_configuration(const TypeARTConfigInit& init) {
  auto file_opts = init.mode != TypeARTConfigInit::FileConfigurationMode::Empty
                       ? config::file::make_file_configuration(init.file_path)
                       : config::file::make_default_file_configuration();
  if (file_opts) {
    auto cl_opts  = std::make_unique<config::cl::CommandLineOptions>();
    auto env_opts = std::make_unique<config::env::EnvironmentFlagsOptions>();
    auto config   = std::make_unique<config::TypeARTConfiguration>(std::move(file_opts.get()), std::move(cl_opts),
                                                                 std::move(env_opts));
    config->prioritizeCommandline(true);
    return config;
  }
  LOG_FATAL("Could not initialize file configuration: \'" << init.file_path
                                                          << "\'. Reason: " << file_opts.getError().message())
  return file_opts.getError();
}

}  // namespace typeart::config

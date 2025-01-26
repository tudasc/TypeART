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
#include "configuration/Configuration.h"
#include "configuration/EnvironmentConfiguration.h"
#include "configuration/FileConfiguration.h"
#include "configuration/TypeARTOptions.h"
#include "support/Logger.h"

#include <string>
#include <string_view>

namespace typeart::config {

TypeARTConfiguration::TypeARTConfiguration(std::unique_ptr<file::FileOptions> config_options,
                                           std::unique_ptr<cl::CommandLineOptions> commandline_options,
                                           std::unique_ptr<env::EnvironmentFlagsOptions> env_options)
    : configuration_options_(std::move(config_options)),
      commandline_options_(std::move(commandline_options)),
      env_options_(std::move(env_options)) {
}

std::optional<OptionValue> TypeARTConfiguration::getValue(std::string_view opt_path) const {
  if (prioritize_commandline) {
    if (auto value = env_options_->getValue(opt_path)) {
      LOG_DEBUG("Take ENV " << opt_path << "=" << (std::string(value.value())));
      return value.value();
    }
    if (auto value = commandline_options_->getValue(opt_path)) {
      LOG_DEBUG("Take CL " << opt_path << "=" << (std::string(value.value())));
      return value.value();
    }
  }
  return configuration_options_->getValue(opt_path);
}

OptionValue TypeARTConfiguration::getValueOr(std::string_view opt_path, OptionValue alt) const {
  if (prioritize_commandline) {
    if (auto value = env_options_->getValue(opt_path)) {
      LOG_DEBUG("Take ENV " << opt_path << "=" << (std::string(value.value())));
      return value.value();
    }
    if (auto value = commandline_options_->getValue(opt_path)) {
      LOG_DEBUG("Take CL " << opt_path << "=" << (std::string(value.value())));
      return value.value();
    }
  }
  return configuration_options_->getValueOr(opt_path, alt);
}

OptionValue TypeARTConfiguration::operator[](std::string_view opt_path) const {
  if (prioritize_commandline) {
    if (auto value = env_options_->getValue(opt_path)) {
      LOG_DEBUG("Take ENV " << opt_path << "=" << (std::string(value.value())));
      return value.value();
    }
    if (auto value = commandline_options_->getValue(opt_path)) {
      LOG_DEBUG("Take CL " << opt_path << "=" << (std::string(value.value())));
      return value.value();
    }
  }
  auto result = configuration_options_->operator[](opt_path);
  LOG_DEBUG("Take File " << opt_path << "=" << (std::string(result)));
  return result;
}

void TypeARTConfiguration::prioritizeCommandline(bool do_prioritize) {
  prioritize_commandline = do_prioritize;
}

void TypeARTConfiguration::emitTypeartFileConfiguration(llvm::raw_ostream& out_stream) {
  out_stream << configuration_options_->getConfigurationAsString();
}

template <typename ClOpt>
std::pair<llvm::StringRef, typename OptionsMap::mapped_type> make_occurr_entry(std::string&& key, ClOpt&& cl_opt) {
  return {key, (cl_opt.getNumOccurrences() > 0)};
}

TypeARTConfigOptions TypeARTConfiguration::getOptions() const {
  return helper::config_to_options(*this);
}

inline llvm::ErrorOr<std::unique_ptr<TypeARTConfiguration>> make_config(
    llvm::ErrorOr<std::unique_ptr<file::FileOptions>> file_opts) {
  if (file_opts) {
    auto cl_opts  = std::make_unique<config::cl::CommandLineOptions>();
    auto env_opts = std::make_unique<config::env::EnvironmentFlagsOptions>();
    auto config   = std::make_unique<config::TypeARTConfiguration>(std::move(file_opts.get()), std::move(cl_opts),
                                                                 std::move(env_opts));
    config->prioritizeCommandline(true);
    return config;
  }
  LOG_FATAL("Could not initialize file configuration. Reason: " << file_opts.getError().message())
  return file_opts.getError();
}

llvm::ErrorOr<std::unique_ptr<TypeARTConfiguration>> make_typeart_configuration(const TypeARTConfigInit& init) {
  auto file_opts = init.mode != TypeARTConfigInit::FileConfigurationMode::Empty
                       ? config::file::make_file_configuration(init.file_path)
                       : config::file::make_default_file_configuration();
  return make_config(std::move(file_opts));
}

llvm::ErrorOr<std::unique_ptr<TypeARTConfiguration>> make_typeart_configuration_from_opts(
    const TypeARTConfigOptions& opts) {
  auto file_opts = config::file::make_from_configuration(opts);
  return make_config(std::move(file_opts));
}

}  // namespace typeart::config

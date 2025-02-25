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

#ifndef TYPEART_TYPEARTCONFIGURATION_H
#define TYPEART_TYPEARTCONFIGURATION_H

#include "configuration/Configuration.h"
#include "configuration/TypeARTOptions.h"

#include "llvm/Support/ErrorOr.h"

namespace typeart::config {

namespace file {
class FileOptions;
}  // namespace file

namespace cl {
class CommandLineOptions;
}  // namespace cl

namespace env {
class EnvironmentFlagsOptions;
}  // namespace env

class TypeARTConfiguration final : public Configuration {
 private:
  std::unique_ptr<file::FileOptions> configuration_options_;
  std::unique_ptr<cl::CommandLineOptions> commandline_options_;
  std::unique_ptr<env::EnvironmentFlagsOptions> env_options_;
  bool prioritize_commandline{true};

 public:
  TypeARTConfiguration(std::unique_ptr<file::FileOptions> config_options,
                       std::unique_ptr<cl::CommandLineOptions> commandline_options,
                       std::unique_ptr<env::EnvironmentFlagsOptions> env_options);
  void prioritizeCommandline(bool do_prioritize);
  [[nodiscard]] std::optional<OptionValue> getValue(std::string_view opt_path) const override;
  [[nodiscard]] OptionValue getValueOr(std::string_view opt_path, const OptionValue& alt) const override;
  [[nodiscard]] OptionValue operator[](std::string_view opt_path) const override;
  void emitTypeartFileConfiguration(llvm::raw_ostream& out_stream);
  [[nodiscard]] TypeARTConfigOptions getOptions() const;

  ~TypeARTConfiguration() override = default;
};

struct TypeARTConfigInit {
  enum class FileConfigurationMode { Empty, Yaml };
  std::string file_path{};
  FileConfigurationMode mode{FileConfigurationMode::Yaml};
};

[[maybe_unused]] llvm::ErrorOr<std::unique_ptr<TypeARTConfiguration>> make_typeart_configuration(
    const TypeARTConfigInit& init);

[[maybe_unused]] llvm::ErrorOr<std::unique_ptr<TypeARTConfiguration>> make_typeart_configuration_from_opts(
    const TypeARTConfigOptions& opts);

}  // namespace typeart::config

#endif  // TYPEART_TYPEARTCONFIGURATION_H

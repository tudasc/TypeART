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

#include "FileConfiguration.h"

#include "Configuration.h"
#include "TypeARTConfiguration.h"
#include "TypeARTOptions.h"
#include "analysis/MemInstFinder.h"
#include "typegen/TypeGenerator.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/YAMLTraits.h"

#include <optional>
#include <string_view>
#include <type_traits>

using namespace llvm;

namespace typeart::config::file {

std::string write_file_configuration_as_text(const FileOptions& file_options);

class YamlFileConfiguration final : public FileOptions {
 private:
  OptionsMap mapping_;

 public:
  explicit YamlFileConfiguration(const TypeARTConfigOptions& conf_file);

  [[nodiscard]] std::optional<config::OptionValue> getValue(std::string_view opt_path) const override;

  [[nodiscard]] OptionsMap getConfiguration() const override;

  [[nodiscard]] std::string getConfigurationAsString() const override;

  ~YamlFileConfiguration() override = default;
};

YamlFileConfiguration::YamlFileConfiguration(const TypeARTConfigOptions& conf_file)
    : mapping_(helper::options_to_map(conf_file)) {
}

std::optional<typeart::config::OptionValue> YamlFileConfiguration::getValue(std::string_view opt_path) const {
  auto key = llvm::StringRef(opt_path.data());
  if (mapping_.count(key) != 0U) {
    return mapping_.lookup(key);
  }
  return {};
}

OptionsMap YamlFileConfiguration::getConfiguration() const {
  return this->mapping_;
}

std::string YamlFileConfiguration::getConfigurationAsString() const {
  return write_file_configuration_as_text(*this);
}

namespace compat {
auto open_flag() {
#if LLVM_VERSION_MAJOR < 13
  return llvm::sys::fs::OpenFlags::F_Text;
#else
  return llvm::sys::fs::OpenFlags::OF_Text;
#endif
}
}  // namespace compat

[[maybe_unused]] llvm::ErrorOr<std::unique_ptr<FileOptions>> make_file_configuration(std::string_view file_path) {
  using namespace llvm;

  ErrorOr<std::unique_ptr<MemoryBuffer>> memBuffer = MemoryBuffer::getFile(file_path.data());

  if (std::error_code error = memBuffer.getError(); error) {
    LOG_WARNING("Warning while loading configuration file \'" << file_path.data() << "\'. Reason: " << error.message());
    return error;
  }

  llvm::yaml::Input input_yaml(memBuffer.get()->getMemBufferRef());
  const auto file_content = io::yaml::yaml_read_file(input_yaml);

  return std::make_unique<YamlFileConfiguration>(file_content);
}

llvm::ErrorOr<std::unique_ptr<FileOptions>> make_default_file_configuration() {
  TypeARTConfigOptions options;
  return std::make_unique<YamlFileConfiguration>(options);
}

llvm::ErrorOr<std::unique_ptr<FileOptions>> make_from_configuration(const TypeARTConfigOptions& options) {
  return std::make_unique<YamlFileConfiguration>(options);
}

llvm::ErrorOr<bool> write_file_configuration(llvm::raw_ostream& out_stream, const FileOptions& options) {
  using namespace llvm;

  llvm::yaml::Output out(out_stream);

  auto data = options.getConfiguration();

  auto conf_file = helper::map_to_options(options.getConfiguration());
  io::yaml::yaml_output_file(out, conf_file);

  return true;
}

std::string write_file_configuration_as_text(const FileOptions& file_options) {
  std::string config_text;
  llvm::raw_string_ostream sstream(config_text);
  if (!write_file_configuration(sstream, file_options)) {
    LOG_WARNING("Could not write config file to string stream.")
  }
  return sstream.str();
}

}  // namespace typeart::config::file

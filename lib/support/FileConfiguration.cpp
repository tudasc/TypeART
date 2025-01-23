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

#include "TypeARTConfiguration.h"
#include "TypeARTOptions.h"
#include "analysis/MemInstFinder.h"
#include "support/Configuration.h"
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

using typeart::config::ConfigStdArgValues;

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
    : mapping_(config::options_to_map(conf_file)) {
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

namespace yaml {
TypeARTConfigOptions yaml_read_file(llvm::yaml::Input& input);
void yaml_output_file(llvm::yaml::Output& output, TypeARTConfigOptions& config);
}  // namespace yaml

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
  const auto file_content = typeart::config::file::yaml::yaml_read_file(input_yaml);

  return std::make_unique<YamlFileConfiguration>(file_content);
}

llvm::ErrorOr<std::unique_ptr<FileOptions>> make_default_file_configuration() {
  TypeARTConfigOptions options;
  return std::make_unique<YamlFileConfiguration>(options);
}

llvm::ErrorOr<std::unique_ptr<FileOptions>> make_from_configuration(const TypeARTConfigOptions& options) {
  return std::make_unique<YamlFileConfiguration>(options);
}

llvm::ErrorOr<bool> write_file_configuration(llvm::raw_ostream& oss, const FileOptions& options) {
  using namespace llvm;

  llvm::yaml::Output out(oss);

  auto data = options.getConfiguration();

  auto conf_file = map_to_options(options.getConfiguration());
  yaml::yaml_output_file(out, conf_file);

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

using namespace llvm::yaml;
using namespace typeart::config::file;

template <>
struct ScalarEnumerationTraits<typeart::analysis::FilterImplementation> {
  static void enumeration(IO& io, typeart::analysis::FilterImplementation& value) {
    io.enumCase(value, "cg", typeart::analysis::FilterImplementation::cg);
    io.enumCase(value, "std", typeart::analysis::FilterImplementation::standard);
    io.enumCase(value, "none", typeart::analysis::FilterImplementation::none);
  }
};

template <>
struct ScalarEnumerationTraits<typeart::TypegenImplementation> {
  static void enumeration(IO& io, typeart::TypegenImplementation& value) {
    io.enumCase(value, "dimeta", typeart::TypegenImplementation::DIMETA);
    io.enumCase(value, "ir", typeart::TypegenImplementation::IR);
  }
};

template <>
struct llvm::yaml::MappingTraits<typeart::config::TypeARTAnalysisOptions> {
  static void mapping(IO& yml_io, typeart::config::TypeARTAnalysisOptions& info) {
    using typeart::config::ConfigStdArgs;
    const auto drop_prefix = [](const std::string& path, std::string_view prefix = "analysis-") {
      llvm::StringRef prefix_less{path};
      prefix_less.consume_front(prefix.data());
      return prefix_less;
    };
    yml_io.mapOptional(drop_prefix(ConfigStdArgs::analysis_filter_global).data(), info.filter_global);
    yml_io.mapOptional(drop_prefix(ConfigStdArgs::analysis_filter_heap_alloc).data(), info.filter_heap_alloc);
    yml_io.mapOptional(drop_prefix(ConfigStdArgs::analysis_filter_pointer_alloc).data(), info.filter_pointer_alloc);
    yml_io.mapOptional(drop_prefix(ConfigStdArgs::analysis_filter_alloca_non_array).data(),
                       info.filter_alloca_non_array);
  }
};

template <>
struct llvm::yaml::MappingTraits<typeart::config::TypeARTCallFilterOptions> {
  static void mapping(IO& yml_io, typeart::config::TypeARTCallFilterOptions& info) {
    using typeart::config::ConfigStdArgs;
    const auto drop_prefix = [](const std::string& path, std::string_view prefix = "filter-") {
      llvm::StringRef prefix_less{path};
      prefix_less.consume_front(prefix.data());
      return prefix_less;
    };
    yml_io.mapOptional(drop_prefix(ConfigStdArgs::filter_impl).data(), info.implementation);
    yml_io.mapOptional(drop_prefix(ConfigStdArgs::filter_glob).data(), info.glob);
    yml_io.mapOptional(drop_prefix(ConfigStdArgs::filter_glob_deep).data(), info.glob_deep);
    yml_io.mapOptional(drop_prefix(ConfigStdArgs::filter_cg_file).data(), info.cg_file);
  }
};

template <>
struct llvm::yaml::MappingTraits<typeart::config::TypeARTConfigOptions> {
  static void mapping(IO& yml_io, typeart::config::TypeARTConfigOptions& info) {
    using typeart::config::ConfigStdArgs;
    yml_io.mapRequired(ConfigStdArgs::types, info.types);
    yml_io.mapRequired(ConfigStdArgs::heap, info.heap);
    yml_io.mapRequired(ConfigStdArgs::stack, info.stack);
    yml_io.mapOptional(ConfigStdArgs::global, info.global);
    yml_io.mapOptional(ConfigStdArgs::stats, info.statistics);
    yml_io.mapOptional(ConfigStdArgs::stack_lifetime, info.stack_lifetime);
    yml_io.mapRequired(ConfigStdArgs::typegen, info.typegen);
    yml_io.mapRequired(ConfigStdArgs::filter, info.filter);
    yml_io.mapOptional("call-filter", info.call_filter_configuration);
    yml_io.mapOptional("analysis", info.analysis_configuration);
    // yml_io.mapOptional("file-format", info.version);
  }
};

namespace typeart::config::file::yaml {

TypeARTConfigOptions yaml_read_file(llvm::yaml::Input& input) {
  TypeARTConfigOptions file_content{};
  input >> file_content;

  return file_content;
}

void yaml_output_file(llvm::yaml::Output& output, TypeARTConfigOptions& config) {
  output << config;
}

}  // namespace typeart::config::file::yaml

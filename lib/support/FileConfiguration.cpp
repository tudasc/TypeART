// TypeART library
//
// Copyright (c) 2017-2022 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#include "FileConfiguration.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/YAMLTraits.h"

using namespace llvm;

namespace typeart::config::file {

using typeart::config::ConfigStdArgValues;

struct ConfigurationOptions {
  std::string types{ConfigStdArgValues::types};
  bool heap{ConfigStdArgValues::heap};
  bool stack{ConfigStdArgValues::stack};
  bool global{ConfigStdArgValues::global};
  bool statistics{ConfigStdArgValues::stats};
  bool stack_lifetime{ConfigStdArgValues::stack_lifetime};
  bool filter{true};
  struct CallFilter {
    std::string implementation{ConfigStdArgValues::filter_impl};
    std::string glob{ConfigStdArgValues::filter_glob};
    std::string glob_deep{ConfigStdArgValues::filter_glob_deep};
    std::string cg_file{ConfigStdArgValues::filter_cg_file};
  };
  CallFilter call_filter_configuration{};
  struct Analysis {
    bool filter_global{ConfigStdArgValues::analysis_filter_global};
    bool filter_heap_alloc{ConfigStdArgValues::analysis_filter_heap_alloc};
    bool filter_pointer_alloc{ConfigStdArgValues::analysis_filter_pointer_alloc};
    bool filter_alloca_non_array{ConfigStdArgValues::analysis_filter_alloca_non_array};
  };
  Analysis analysis_configuration{};
  int version{1};
};

namespace helper {

ConfigurationOptions map2config(const FileOptionsMap& mapping) {
  const auto make_entry = [&mapping](std::string_view entry, auto& ref) {
    auto key = llvm::StringRef(entry.data());
    ref      = static_cast<typename std::remove_reference<decltype(ref)>::type>(mapping.lookup(key));
  };

  ConfigurationOptions conf_file;
  make_entry(ConfigStdArgs::types, conf_file.types);
  make_entry(ConfigStdArgs::stats, conf_file.statistics);
  make_entry(ConfigStdArgs::heap, conf_file.heap);
  make_entry(ConfigStdArgs::global, conf_file.global);
  make_entry(ConfigStdArgs::stack, conf_file.stack);
  make_entry(ConfigStdArgs::stack_lifetime, conf_file.stack_lifetime);
  make_entry(ConfigStdArgs::filter, conf_file.filter);
  make_entry(ConfigStdArgs::filter_impl, conf_file.call_filter_configuration.implementation);
  make_entry(ConfigStdArgs::filter_glob, conf_file.call_filter_configuration.glob);
  make_entry(ConfigStdArgs::filter_glob_deep, conf_file.call_filter_configuration.glob_deep);
  make_entry(ConfigStdArgs::filter_cg_file, conf_file.call_filter_configuration.cg_file);
  make_entry(ConfigStdArgs::analysis_filter_global, conf_file.analysis_configuration.filter_global);
  make_entry(ConfigStdArgs::analysis_filter_heap_alloc, conf_file.analysis_configuration.filter_heap_alloc);
  make_entry(ConfigStdArgs::analysis_filter_pointer_alloc, conf_file.analysis_configuration.filter_pointer_alloc);
  make_entry(ConfigStdArgs::analysis_filter_alloca_non_array, conf_file.analysis_configuration.filter_alloca_non_array);
  return conf_file;
}

FileOptionsMap config2map(const ConfigurationOptions& conf_file) {
  using namespace detail;

  const auto make_entry = [](std::string&& key,
                             const auto& field_value) -> std::pair<StringRef, typename FileOptionsMap ::mapped_type> {
    LOG_DEBUG(key << "->" << field_value)
    return {key, config::OptionValue{field_value}};
  };
  FileOptionsMap mapping_ = {
      make_entry(ConfigStdArgs::types, conf_file.types),
      make_entry(ConfigStdArgs::stats, conf_file.statistics),
      make_entry(ConfigStdArgs::heap, conf_file.heap),
      make_entry(ConfigStdArgs::global, conf_file.global),
      make_entry(ConfigStdArgs::stack, conf_file.stack),
      make_entry(ConfigStdArgs::stack_lifetime, conf_file.stack_lifetime),
      make_entry(ConfigStdArgs::filter, conf_file.filter),
      make_entry(ConfigStdArgs::filter_impl, conf_file.call_filter_configuration.implementation),
      make_entry(ConfigStdArgs::filter_glob, conf_file.call_filter_configuration.glob),
      make_entry(ConfigStdArgs::filter_glob_deep, conf_file.call_filter_configuration.glob_deep),
      make_entry(ConfigStdArgs::filter_cg_file, conf_file.call_filter_configuration.cg_file),
      make_entry(ConfigStdArgs::analysis_filter_global, conf_file.analysis_configuration.filter_global),
      make_entry(ConfigStdArgs::analysis_filter_heap_alloc, conf_file.analysis_configuration.filter_heap_alloc),
      make_entry(ConfigStdArgs::analysis_filter_pointer_alloc, conf_file.analysis_configuration.filter_pointer_alloc),
      make_entry(ConfigStdArgs::analysis_filter_alloca_non_array,
                 conf_file.analysis_configuration.filter_alloca_non_array),
  };
  return mapping_;
}

}  // namespace helper

std::string write_file_configuration_as_text(const FileOptions& file_options);

class YamlFileConfiguration final : public FileOptions {
 private:
  FileOptionsMap mapping_;

 public:
  explicit YamlFileConfiguration(const ConfigurationOptions& conf_file);

  [[nodiscard]] Optional<config::OptionValue> getValue(std::string_view opt_path) const override;

  [[nodiscard]] FileOptionsMap getConfiguration() const override;

  [[nodiscard]] std::string getConfigurationAsString() const override;

  ~YamlFileConfiguration() override = default;
};

YamlFileConfiguration::YamlFileConfiguration(const ConfigurationOptions& conf_file)
    : mapping_(helper::config2map(conf_file)) {
}

llvm::Optional<typeart::config::OptionValue> YamlFileConfiguration::getValue(std::string_view opt_path) const {
  auto key = llvm::StringRef(opt_path.data());
  if (mapping_.count(key) != 0U) {
    return mapping_.lookup(key);
  }
  return llvm::None;
}

FileOptionsMap YamlFileConfiguration::getConfiguration() const {
  return this->mapping_;
}

std::string YamlFileConfiguration::getConfigurationAsString() const {
  return write_file_configuration_as_text(*this);
}

namespace yaml {
ConfigurationOptions yaml_read_file(llvm::yaml::Input& input);
void yaml_output_file(llvm::yaml::Output& output, ConfigurationOptions& config);
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
  ConfigurationOptions options{};
  return std::make_unique<YamlFileConfiguration>(options);
}

llvm::ErrorOr<bool> write_file_configuration(llvm::raw_ostream& oss, const FileOptions& options) {
  using namespace llvm;

  llvm::yaml::Output out(oss);

  auto data = options.getConfiguration();

  auto conf_file = helper::map2config(options.getConfiguration());
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
struct llvm::yaml::MappingTraits<typeart::config::file::ConfigurationOptions::Analysis> {
  static void mapping(IO& yml_io, typeart::config::file::ConfigurationOptions::Analysis& info) {
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
struct llvm::yaml::MappingTraits<typeart::config::file::ConfigurationOptions::CallFilter> {
  static void mapping(IO& yml_io, typeart::config::file::ConfigurationOptions::CallFilter& info) {
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
struct llvm::yaml::MappingTraits<typeart::config::file::ConfigurationOptions> {
  static void mapping(IO& yml_io, typeart::config::file::ConfigurationOptions& info) {
    using typeart::config::ConfigStdArgs;
    yml_io.mapRequired(ConfigStdArgs::types, info.types);
    yml_io.mapRequired(ConfigStdArgs::heap, info.heap);
    yml_io.mapRequired(ConfigStdArgs::stack, info.stack);
    yml_io.mapOptional(ConfigStdArgs::global, info.global);
    yml_io.mapOptional(ConfigStdArgs::stats, info.statistics);
    yml_io.mapOptional(ConfigStdArgs::stack_lifetime, info.stack_lifetime);
    yml_io.mapRequired(ConfigStdArgs::filter, info.filter);
    yml_io.mapOptional("call-filter", info.call_filter_configuration);
    yml_io.mapOptional("analysis", info.analysis_configuration);
    yml_io.mapOptional("file-format", info.version);
  }
};

namespace typeart::config::file::yaml {

ConfigurationOptions yaml_read_file(llvm::yaml::Input& input) {
  ConfigurationOptions file_content{};
  input >> file_content;

  return file_content;
}

void yaml_output_file(llvm::yaml::Output& output, ConfigurationOptions& config) {
  output << config;
}

}  // namespace typeart::config::file::yaml

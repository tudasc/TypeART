

#include "TypeARTOptions.h"

#include "FileConfiguration.h"
#include "support/Configuration.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/YAMLTraits.h"

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
    yml_io.mapOptional("call-filter", info.filter_config);
    yml_io.mapOptional("analysis", info.analysis_config);
    // yml_io.mapOptional("file-format", info.version);
  }
};

namespace typeart::config::io::yaml {

TypeARTConfigOptions yaml_read_file(llvm::yaml::Input& input) {
  TypeARTConfigOptions file_content{};
  input >> file_content;

  return file_content;
}

void yaml_output_file(llvm::yaml::Output& output, const TypeARTConfigOptions& config) {
  output << const_cast<TypeARTConfigOptions&>(config);
}

}  // namespace typeart::config::io::yaml

namespace typeart::config {
  
llvm::raw_ostream& operator<<(llvm::raw_ostream& out_stream, const TypeARTConfigOptions& options) {
  std::string config_text;
  llvm::raw_string_ostream sstream(config_text);
  llvm::yaml::Output out(sstream);
  io::yaml::yaml_output_file(out, options);
  out_stream.flush();
  out_stream << sstream.str();
  return out_stream;
}

namespace helper {

template <typename Constructor>
TypeARTConfigOptions construct_with(Constructor&& make_entry) {
  TypeARTConfigOptions config;
  make_entry(ConfigStdArgs::types, config.types);
  make_entry(ConfigStdArgs::stats, config.statistics);
  make_entry(ConfigStdArgs::heap, config.heap);
  make_entry(ConfigStdArgs::global, config.global);
  make_entry(ConfigStdArgs::stack, config.stack);
  make_entry(ConfigStdArgs::stack_lifetime, config.stack_lifetime);
  make_entry(ConfigStdArgs::filter, config.filter);
  make_entry(ConfigStdArgs::filter_impl, config.filter_config.implementation);
  make_entry(ConfigStdArgs::filter_glob, config.filter_config.glob);
  make_entry(ConfigStdArgs::filter_glob_deep, config.filter_config.glob_deep);
  make_entry(ConfigStdArgs::filter_cg_file, config.filter_config.cg_file);
  make_entry(ConfigStdArgs::analysis_filter_global, config.analysis_config.filter_global);
  make_entry(ConfigStdArgs::analysis_filter_heap_alloc, config.analysis_config.filter_heap_alloc);
  make_entry(ConfigStdArgs::analysis_filter_pointer_alloc, config.analysis_config.filter_pointer_alloc);
  make_entry(ConfigStdArgs::analysis_filter_alloca_non_array, config.analysis_config.filter_alloca_non_array);
  make_entry(ConfigStdArgs::typegen, config.typegen);
  return config;
}

TypeARTConfigOptions map_to_options(const OptionsMap& mapping) {
  const auto make_entry = [&](std::string_view entry, auto& ref) {
    auto key = llvm::StringRef(entry.data());
    ref      = static_cast<typename std::remove_reference<decltype(ref)>::type>(mapping.lookup(key));
  };
  return construct_with(make_entry);
}

TypeARTConfigOptions config_to_options(const Configuration& configuration) {
  const auto make_entry = [&](std::string_view entry, auto& ref) {
    auto key = llvm::StringRef(entry.data());
    ref      = static_cast<typename std::remove_reference<decltype(ref)>::type>(configuration[key]);
  };
  return construct_with(make_entry);
}

template <typename T>
auto make_entry(std::string_view key, const T& field_value)
    -> std::pair<llvm::StringRef, typename OptionsMap::mapped_type> {
  if constexpr (std::is_enum_v<T>) {
    return {key, config::OptionValue{static_cast<int>(field_value)}};
  } else {
    return {key, config::OptionValue{field_value}};
  }
};

OptionsMap options_to_map(const TypeARTConfigOptions& config) {
  OptionsMap mapping_ = {
      make_entry(ConfigStdArgs::types, config.types),
      make_entry(ConfigStdArgs::stats, config.statistics),
      make_entry(ConfigStdArgs::heap, config.heap),
      make_entry(ConfigStdArgs::global, config.global),
      make_entry(ConfigStdArgs::stack, config.stack),
      make_entry(ConfigStdArgs::stack_lifetime, config.stack_lifetime),
      make_entry(ConfigStdArgs::typegen, config.typegen),
      make_entry(ConfigStdArgs::filter, config.filter),
      make_entry(ConfigStdArgs::filter_impl, config.filter_config.implementation),
      make_entry(ConfigStdArgs::filter_glob, config.filter_config.glob),
      make_entry(ConfigStdArgs::filter_glob_deep, config.filter_config.glob_deep),
      make_entry(ConfigStdArgs::filter_cg_file, config.filter_config.cg_file),
      make_entry(ConfigStdArgs::analysis_filter_global, config.analysis_config.filter_global),
      make_entry(ConfigStdArgs::analysis_filter_heap_alloc, config.analysis_config.filter_heap_alloc),
      make_entry(ConfigStdArgs::analysis_filter_pointer_alloc, config.analysis_config.filter_pointer_alloc),
      make_entry(ConfigStdArgs::analysis_filter_alloca_non_array, config.analysis_config.filter_alloca_non_array),
  };
  return mapping_;
}

}  // namespace helper

}  // namespace typeart::config

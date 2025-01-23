

#include "TypeARTOptions.h"

#include "support/Configuration.h"

namespace typeart::config {

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
  make_entry(ConfigStdArgs::filter_impl, config.call_filter_configuration.implementation);
  make_entry(ConfigStdArgs::filter_glob, config.call_filter_configuration.glob);
  make_entry(ConfigStdArgs::filter_glob_deep, config.call_filter_configuration.glob_deep);
  make_entry(ConfigStdArgs::filter_cg_file, config.call_filter_configuration.cg_file);
  make_entry(ConfigStdArgs::analysis_filter_global, config.analysis_configuration.filter_global);
  make_entry(ConfigStdArgs::analysis_filter_heap_alloc, config.analysis_configuration.filter_heap_alloc);
  make_entry(ConfigStdArgs::analysis_filter_pointer_alloc, config.analysis_configuration.filter_pointer_alloc);
  make_entry(ConfigStdArgs::analysis_filter_alloca_non_array, config.analysis_configuration.filter_alloca_non_array);
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
      make_entry(ConfigStdArgs::filter_impl, config.call_filter_configuration.implementation),
      make_entry(ConfigStdArgs::filter_glob, config.call_filter_configuration.glob),
      make_entry(ConfigStdArgs::filter_glob_deep, config.call_filter_configuration.glob_deep),
      make_entry(ConfigStdArgs::filter_cg_file, config.call_filter_configuration.cg_file),
      make_entry(ConfigStdArgs::analysis_filter_global, config.analysis_configuration.filter_global),
      make_entry(ConfigStdArgs::analysis_filter_heap_alloc, config.analysis_configuration.filter_heap_alloc),
      make_entry(ConfigStdArgs::analysis_filter_pointer_alloc, config.analysis_configuration.filter_pointer_alloc),
      make_entry(ConfigStdArgs::analysis_filter_alloca_non_array,
                 config.analysis_configuration.filter_alloca_non_array),
  };
  return mapping_;
}

}  // namespace typeart::config
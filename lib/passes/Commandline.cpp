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

#include "Commandline.h"

#include "analysis/MemInstFinder.h"
#include "support/Logger.h"

#include "llvm/Support/CommandLine.h"

using namespace llvm;

namespace typeart::config::cl {
struct CommandlineStdArgs final {
#define TYPEART_CONFIG_OPTION(name, path, type, def_value, description) static constexpr char name[] = "typeart-" path;
#include "support/ConfigurationBaseOptions.h"
#undef TYPEART_CONFIG_OPTION
};
}  // namespace typeart::config::cl

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
               clEnumValN(typeart::analysis::FilterImplementation::cg, "cg", "Call-graph-based filter"),
               clEnumValN(typeart::analysis::FilterImplementation::acg, "acg",
                          "Call-graph-based filter (distinguishes arguments)")),
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

std::string get_type_file_path() {
  if (!cl_typeart_type_file.empty()) {
    LOG_DEBUG("Using cl::opt for types file " << cl_typeart_type_file.getValue());
    return cl_typeart_type_file.getValue();
  }
  const char* type_file = std::getenv("TYPEART_TYPE_FILE");
  if (type_file != nullptr) {
    LOG_DEBUG("Using env var for types file " << type_file)
    return std::string{type_file};
  }
  LOG_DEBUG("Loading default types file types.yaml");
  return "types.yaml";
}

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

  mapping_ = {
      make_entry(ConfigStdArgs::types, get_type_file_path()),
      make_entry(ConfigStdArgs::stats, cl_typeart_stats),
      make_entry(ConfigStdArgs::heap, cl_typeart_instrument_heap),
      make_entry(ConfigStdArgs::global, cl_typeart_instrument_global),
      make_entry(ConfigStdArgs::stack, cl_typeart_instrument_stack),
      make_entry(ConfigStdArgs::stack_lifetime, cl_typeart_instrument_stack_lifetime),
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

llvm::Optional<typeart::config::OptionValue> CommandLineOptions::getValue(std::string_view opt_path) const {
  auto key = llvm::StringRef(opt_path.data());
  if (mapping_.count(key) != 0U) {
    return mapping_.lookup(key);
  }
  return llvm::None;
}

[[maybe_unused]] bool CommandLineOptions::valueSpecified(std::string_view opt_path) const {
  auto key = llvm::StringRef(opt_path.data());
  return occurence_mapping_.lookup(key);
}

}  // namespace typeart::config::cl

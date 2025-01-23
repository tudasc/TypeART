#ifndef C86BA97A_734C_4A62_A56E_9E38A9E55DE6
#define C86BA97A_734C_4A62_A56E_9E38A9E55DE6

#include "../passes/analysis/MemInstFinder.h"
#include "../passes/typegen/TypeGenerator.h"
#include "support/Configuration.h"

namespace llvm::yaml {
class Input;
class Output;
}  // namespace llvm::yaml

namespace typeart::config {

using typeart::analysis::FilterImplementation;
using typeart::config::ConfigStdArgValues;

struct TypeARTCallFilterOptions {
  FilterImplementation implementation{FilterImplementation::standard};
  std::string glob{ConfigStdArgValues::filter_glob};
  std::string glob_deep{ConfigStdArgValues::filter_glob_deep};
  std::string cg_file{ConfigStdArgValues::filter_cg_file};
};

struct TypeARTAnalysisOptions {
  bool filter_global{ConfigStdArgValues::analysis_filter_global};
  bool filter_heap_alloc{ConfigStdArgValues::analysis_filter_heap_alloc};
  bool filter_pointer_alloc{ConfigStdArgValues::analysis_filter_pointer_alloc};
  bool filter_alloca_non_array{ConfigStdArgValues::analysis_filter_alloca_non_array};
};

struct TypeARTConfigOptions {
  std::string types{ConfigStdArgValues::types};
  bool heap{ConfigStdArgValues::heap};
  bool stack{ConfigStdArgValues::stack};
  bool global{ConfigStdArgValues::global};
  bool statistics{ConfigStdArgValues::stats};
  bool stack_lifetime{ConfigStdArgValues::stack_lifetime};
  TypegenImplementation typegen{TypegenImplementation::DIMETA};
  bool filter{false};

  TypeARTCallFilterOptions filter_config{};
  TypeARTAnalysisOptions analysis_config{};
};

namespace helper {
TypeARTConfigOptions map_to_options(const OptionsMap&);

TypeARTConfigOptions config_to_options(const Configuration&);

OptionsMap options_to_map(const TypeARTConfigOptions&);
}  // namespace helper

namespace io::yaml {

TypeARTConfigOptions yaml_read_file(llvm::yaml::Input& input);

void yaml_output_file(llvm::yaml::Output& output, const TypeARTConfigOptions& config);

}  // namespace io::yaml

llvm::raw_ostream& operator<<(llvm::raw_ostream& out_s, const TypeARTConfigOptions& options);

}  // namespace typeart::config

#endif /* C86BA97A_734C_4A62_A56E_9E38A9E55DE6 */

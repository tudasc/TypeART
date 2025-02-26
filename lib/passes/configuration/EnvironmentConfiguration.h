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

#ifndef TYPEART_ENVIRONMENT_CONFIGURATION_H
#define TYPEART_ENVIRONMENT_CONFIGURATION_H

#include "configuration/Configuration.h"

namespace typeart::config::env {

struct Phase {
  bool heap;
  bool stack;
};

std::optional<std::string> get_env_flag(std::string_view flag);

class EnvironmentFlagsOptions final : public config::Configuration {
 private:
  OptionsMap mapping_;
  OptOccurrenceMap occurence_mapping_;

 public:
  EnvironmentFlagsOptions();
  [[nodiscard]] std::optional<config::OptionValue> getValue(std::string_view opt_path) const override;
  [[maybe_unused]] [[nodiscard]] bool valueSpecified(std::string_view opt_path) const;
  void parsePhaseEnvFlags(const Phase& phase);
};

}  // namespace typeart::config::env

#endif

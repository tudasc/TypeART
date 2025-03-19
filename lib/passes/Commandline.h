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

#ifndef TYPEART_COMMANDLINE_H
#define TYPEART_COMMANDLINE_H

#include "configuration/Configuration.h"

namespace typeart::config::cl {

class CommandLineOptions final : public config::Configuration {
 private:
  OptionsMap mapping_;
  OptOccurrenceMap occurence_mapping_;

 public:
  CommandLineOptions();
  [[nodiscard]] std::optional<config::OptionValue> getValue(std::string_view opt_path) const override;
  [[maybe_unused]] [[nodiscard]] bool valueSpecified(std::string_view opt_path) const;
};

}  // namespace typeart::config::cl

#endif  // TYPEART_COMMANDLINE_H

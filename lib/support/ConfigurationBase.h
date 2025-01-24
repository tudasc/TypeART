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

#ifndef TYPEART_CONFIG_OPTION_BASE_H
#define TYPEART_CONFIG_OPTION_BASE_H

#include <string>

namespace typeart::config {

struct ConfigStdArgs final {
#define TYPEART_CONFIG_OPTION(name, path, type, def_value, description, upper_path) static constexpr char name[] = path;
#include "ConfigurationBaseOptions.h"
#undef TYPEART_CONFIG_OPTION
};

struct ConfigStdArgValues final {
#define TYPEART_CONFIG_OPTION(name, path, type, def_value, description, upper_path) \
  static constexpr decltype(def_value) name{def_value};
#include "ConfigurationBaseOptions.h"
#undef TYPEART_CONFIG_OPTION
};

struct ConfigStdArgTypes final {
#define TYPEART_CONFIG_OPTION(name, path, type, default_value, description, upper_path) using name##_ty = type;
#include "ConfigurationBaseOptions.h"
#undef TYPEART_CONFIG_OPTION
};

struct ConfigStdArgDescriptions final {
#define TYPEART_CONFIG_OPTION(name, path, type, default_value, description, upper_path) \
  static constexpr char name[] = description;
#include "ConfigurationBaseOptions.h"
#undef TYPEART_CONFIG_OPTION
};

}  // namespace typeart::config

#endif /* TYPEART_CONFIG_OPTION_BASE_H */

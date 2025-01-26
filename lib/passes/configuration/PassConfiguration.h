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

#ifndef TYPEART_PASS_CONFIGURATION_H
#define TYPEART_PASS_CONFIGURATION_H

#include "TypeARTOptions.h"
#include "Configuration.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace typeart::config::pass {

using PassConfig = std::pair<llvm::Expected<TypeARTConfigOptions>, OptOccurrenceMap>;

llvm::Expected<TypeARTConfigOptions> parse_typeart_config(llvm::StringRef parameters);
PassConfig parse_typeart_config_with_occurrence(llvm::StringRef parameters);

}  // namespace typeart::config::pass

#endif /* TYPEART_PASS_CONFIGURATION_H */

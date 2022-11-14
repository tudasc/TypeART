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

#ifndef TYPEART_COMMANDLINE_H
#define TYPEART_COMMANDLINE_H

#include "analysis/MemInstFinder.h"

namespace typeart::cl {

analysis::MemInstFinderConfig get_meminstfinder_configuration();

std::string get_type_file_path();
bool get_instrument_heap();
bool get_instrument_global();
bool get_instrument_stack();
bool get_instrument_stack_lifetime();
bool get_print_stats();

}  // namespace typeart::cl

#endif  // TYPEART_COMMANDLINE_H

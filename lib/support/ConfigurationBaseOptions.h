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

#ifndef TYPEART_CONFIG_OPTION
#define TYPEART_CONFIG_OPTION(name, path, type, default_value, description)
#endif

TYPEART_CONFIG_OPTION(types, "types", std::string, "types.yaml", "Location of the generated type file.")
TYPEART_CONFIG_OPTION(stats, "stats", bool, false, "Show statistics for TypeArt type pass.")
TYPEART_CONFIG_OPTION(heap, "heap", bool, true, "Instrument heap allocation/free instructions.")
TYPEART_CONFIG_OPTION(stack, "stack", bool, false, "Instrument stack allocations.")
TYPEART_CONFIG_OPTION(global, "global", bool, false, "Instrument global allocations.")
TYPEART_CONFIG_OPTION(stack_lifetime, "stack-lifetime", bool, true,
                      "Instrument lifetime.start intrinsic instead of alloca.")
TYPEART_CONFIG_OPTION(filter, "filter", bool, false,
                      "Filter allocas (stack/global) that are passed to relevant function calls.")
TYPEART_CONFIG_OPTION(filter_impl, "filter-implementation", std::string, "std",
                      "Select the call filter implementation.")
TYPEART_CONFIG_OPTION(filter_glob, "filter-glob", std::string, "*MPI_*",
                      "Filter allocas based on the function name (glob) <string>.")
TYPEART_CONFIG_OPTION(filter_glob_deep, "filter-glob-deep", std::string, "MPI_*",
                      "Filter allocas based on specific API: Values passed as void* are correlated when string matched "
                      "and kept when correlated successfully.")
TYPEART_CONFIG_OPTION(filter_cg_file, "filter-cg-file", std::string, "", "Location of call-graph file to use.")
TYPEART_CONFIG_OPTION(analysis_filter_global, "analysis-filter-global", bool, true, "Filter globals of a module.")
TYPEART_CONFIG_OPTION(analysis_filter_heap_alloc, "analysis-filter-heap-alloca", bool, false,
                      "Filter alloca instructions that have a store from a heap allocation.")
TYPEART_CONFIG_OPTION(analysis_filter_pointer_alloc, "analysis-filter-pointer-alloca", bool, true,
                      "Filter allocas of pointer types.")
TYPEART_CONFIG_OPTION(analysis_filter_alloca_non_array, "analysis-filter-non-array-alloca", bool, false,
                      "Filter scalar valued allocas.")
TYPEART_CONFIG_OPTION(typegen, "typegen", std::string, "dimeta", "Select type layout generator.")

#undef TYPEART_CONFIG_OPTION

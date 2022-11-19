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

#ifndef TYPEART_CONFIG_OPTION
#define TYPEART_CONFIG_OPTION(name, path, type, default_value)
#endif

TYPEART_CONFIG_OPTION(types, "types", std::string, "types.yaml")
TYPEART_CONFIG_OPTION(stats, "stats", bool, false)
TYPEART_CONFIG_OPTION(heap, "heap", bool, true)
TYPEART_CONFIG_OPTION(stack, "stack", bool, false)
TYPEART_CONFIG_OPTION(global, "global", bool, false)
TYPEART_CONFIG_OPTION(stack_lifetime, "stack-lifetime", bool, true)
TYPEART_CONFIG_OPTION(filter, "filter", bool, true)
TYPEART_CONFIG_OPTION(filter_impl, "filter-implementation", std::string, "std")
TYPEART_CONFIG_OPTION(filter_glob, "filter-glob", std::string, "*MPI_*")
TYPEART_CONFIG_OPTION(filter_glob_deep, "filter-glob-deep", std::string, "MPI_*")
TYPEART_CONFIG_OPTION(filter_cg_file, "filter-cg-file", std::string, "")
TYPEART_CONFIG_OPTION(analysis_filter_global, "analysis-filter-global", bool, true)
TYPEART_CONFIG_OPTION(analysis_filter_heap_alloc, "analysis-filter-heap-alloc", bool, false)
TYPEART_CONFIG_OPTION(analysis_filter_pointer_alloc, "analysis-filter-pointer-alloc", bool, true)
TYPEART_CONFIG_OPTION(analysis_filter_alloca_non_array, "analysis-filter-alloca-non-array", bool, false)

#undef TYPEART_CONFIG_OPTION

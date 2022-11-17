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

TYPEART_CONFIG_OPTION(types, "typeart:types", std::string, "types.yaml")
TYPEART_CONFIG_OPTION(stats, "typeart:stats", bool, false)
TYPEART_CONFIG_OPTION(heap, "typeart:heap", bool, true)
TYPEART_CONFIG_OPTION(stack, "typeart:stack", bool, false)
TYPEART_CONFIG_OPTION(global, "typeart:global", bool, false)
TYPEART_CONFIG_OPTION(stack_lifetime, "typeart:stack_lifetime", bool, true)
TYPEART_CONFIG_OPTION(filter, "typeart:filter", bool, true)
TYPEART_CONFIG_OPTION(filter_impl, "typeart:filter:implementation", std::string, "std")
TYPEART_CONFIG_OPTION(filter_glob, "typeart:filter:glob", std::string, "*MPI_*")
TYPEART_CONFIG_OPTION(filter_glob_deep, "typeart:filter:glob_deep", std::string, "MPI_*")
TYPEART_CONFIG_OPTION(filter_cg_file, "typeart:filter:cg_file", std::string, "")
TYPEART_CONFIG_OPTION(analysis_filter_global, "typeart:analysis:filter:global", bool, true)
TYPEART_CONFIG_OPTION(analysis_filter_heap_alloc, "typeart:analysis:filter:heap_alloc", bool, false)
TYPEART_CONFIG_OPTION(analysis_filter_pointer_alloc, "typeart:analysis:filter:pointer_alloc", bool, true)
TYPEART_CONFIG_OPTION(analysis_filter_alloca_non_array, "typeart:analysis:filter:alloca_non_array", bool, false)

#undef TYPEART_CONFIG_OPTION

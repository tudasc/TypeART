// TypeART library
//
// Copyright (c) 2017-2021 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#include "Stats.h"
#include "TypeCheck.h"

#include <atomic>

namespace typeart::logger {

void result(const char* name, const void* called_from, bool is_send, const Buffer& buffer, const MPIType& type,
            const Result<void>&);
void error(const char* function_name, const void* called_from, const Error&);
void call_counter(const CallCounter& call_counter, long ru_maxrss);
void mpi_counter(const MPICounter& mpi_counter);
void null_buffer();
void unsupported(const char* name);

}  // namespace typeart::logger

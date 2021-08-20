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

#pragma once

#include "runtime/RuntimeInterface.h"

#include <mpi.h>
#include <optional>
#include <string>

namespace typeart {

const char* error_message_for(typeart_status status);

// Given a builtin MPI type, returns the corresponding TypeArt type.
// If the MPI type is a custom type, -1 is returned.
// Note: this function cannot distinguish between TA_FP128 und TA_PPC_TP128,
// therefore TA_FP128 is always returned in case of an 16 byte floating point
// MPI type. This should be considered by the caller for performing typechecks.
int type_id_for(MPI_Datatype mpi_type);

const char* combiner_name_for(int combiner);

}  // namespace typeart

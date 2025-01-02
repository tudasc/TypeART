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

#ifndef TYPEART_DIMETATYPEGENERATOR_H
#define TYPEART_DIMETATYPEGENERATOR_H

#include "typegen/TypeGenerator.h"

#include <memory>
#include <string_view>

namespace typeart::types {

std::unique_ptr<TypeGenerator> make_dimeta_typeidgen(std::string_view file,
                                                     std::unique_ptr<TypeDatabase> database_of_types);

}  // namespace typeart::types

#endif
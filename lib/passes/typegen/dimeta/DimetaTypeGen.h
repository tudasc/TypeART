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
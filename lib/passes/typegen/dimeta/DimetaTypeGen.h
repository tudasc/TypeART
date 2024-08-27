#ifndef TYPEART_DIMETATYPEGENERATOR_H
#define TYPEART_DIMETATYPEGENERATOR_H

#include "../TypeGenerator.h"

#include <memory>
#include <string_view>

namespace typeart::types {

std::unique_ptr<TypeGenerator> make_dimeta_typeidgen(std::string_view file);

}  // namespace typeart::types

#endif
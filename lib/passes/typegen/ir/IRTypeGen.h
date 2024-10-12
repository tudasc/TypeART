#ifndef TYPEART_IRTYPEGENERATOR_H
#define TYPEART_IRTYPEGENERATOR_H

#include "typegen/TypeGenerator.h"

#include <memory>
#include <string_view>

namespace typeart {

std::unique_ptr<TypeGenerator> make_ir_typeidgen(std::string_view file,
                                                 std::unique_ptr<TypeDatabase> database_of_types);

}

#endif

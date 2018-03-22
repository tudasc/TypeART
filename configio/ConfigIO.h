//
// Created by sebastian on 22.03.18.
//

#ifndef LLVM_MUST_SUPPORT_CONFIGIO_H
#define LLVM_MUST_SUPPORT_CONFIGIO_H

#include <string>

#include "TypeConfig.h"

namespace must
{


class ConfigIO
{
public:
    explicit ConfigIO(TypeConfig* config);

    bool load(std::string file);
    bool store(std::string file) const;

private:
    TypeConfig* config;
};

}

#endif //LLVM_MUST_SUPPORT_CONFIGIO_H

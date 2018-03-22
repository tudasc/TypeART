//
// Created by sebastian on 22.03.18.
//

#include "ConfigIO.h"

#include <fstream>

namespace must
{

ConfigIO::ConfigIO(TypeConfig* config) :
    config(config)
{
}

bool ConfigIO::load(std::string file)
{
    config->clear();
    std::ifstream is;
    is.open(file);
    if (!is.is_open()) {
        return false;
    }
    std::string name;
    int id;
    while (is >> name >> id) {
        config->registerType(name, id);
    }
    is.close();
    return true;
}

bool ConfigIO::store(std::string file) const
{
    std::ofstream os;
    os.open(file);
    if (!os.is_open()) {
        return false;
    }
    for (const auto& typeName : config->getTypeList()) {
        int id = config->getTypeID(typeName);
        os << typeName << " " << id << std::endl;
    }
    os.close();
    return true;
}

}
//
// Created by sebastian on 22.03.18.
//

#include "TypeConfig.h"


namespace must {


TypeConfig::TypeConfig()
{

}

void TypeConfig::clear()
{
    typeMap.clear();
}

void TypeConfig::registerType(std::string typeName, int id)
{
    typeMap.insert({typeName, id});
}

int TypeConfig::getTypeID(std::string typeName) const
{
    auto it = typeMap.find(typeName);
    if (it != typeMap.end()) {
        return it->second;
    }
    return UNDEFINED;
}

std::vector<std::string> TypeConfig::getTypeList() const
{
    std::vector<std::string> typeIDs;
    typeIDs.reserve(typeMap.size());
    for (const auto& entry : typeMap) {
        typeIDs.push_back(entry.first);
    }
    return typeIDs;
}


}
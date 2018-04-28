//
// Created by sebastian on 22.03.18.
//

#ifndef LLVM_MUST_SUPPORT_CONFIGIO_H
#define LLVM_MUST_SUPPORT_CONFIGIO_H

#include <string>

#include "TypeConfig.h"

namespace must {

class ConfigIO {
 public:
  explicit ConfigIO(TypeConfig* config);

  bool load(std::string file);
  bool store(std::string file) const;

 private:
  std::string serialize(StructTypeInfo structInfo) const;
  StructTypeInfo deserialize(std::string infoString) const;

  bool isComment(std::string line) const;

  TypeConfig* config;
};
}

#endif  // LLVM_MUST_SUPPORT_CONFIGIO_H

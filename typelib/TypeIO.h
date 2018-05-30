//
// Created by sebastian on 22.03.18.
//

#ifndef LLVM_MUST_SUPPORT_CONFIGIO_H
#define LLVM_MUST_SUPPORT_CONFIGIO_H

#include <string>

#include "TypeDB.h"

namespace must {

class TypeIO {
 public:
  explicit TypeIO(TypeDB* config);

  bool load(std::string file);
  bool store(std::string file) const;

 private:
  std::string serialize(StructTypeInfo structInfo) const;
  StructTypeInfo deserialize(std::string infoString) const;

  bool isComment(std::string line) const;

  TypeDB* typeDB;
};
}

#endif  // LLVM_MUST_SUPPORT_CONFIGIO_H

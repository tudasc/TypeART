//
// Created by sebastian on 22.03.18.
//

#ifndef LLVM_MUST_SUPPORT_CONFIGIO_H
#define LLVM_MUST_SUPPORT_CONFIGIO_H

#include <string>

namespace typeart {

class TypeDB;

class TypeIO {
 private:
  TypeDB& typeDB;

 public:
  explicit TypeIO(TypeDB& config);
  bool load(const std::string& file);
  bool store(const std::string& file) const;
};

}  // namespace typeart

#endif  // LLVM_MUST_SUPPORT_CONFIGIO_H

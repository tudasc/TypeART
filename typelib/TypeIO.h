//
// Created by sebastian on 22.03.18.
//

#ifndef LLVM_MUST_SUPPORT_CONFIGIO_H
#define LLVM_MUST_SUPPORT_CONFIGIO_H

#include <memory>
#include <string>

namespace must {

class TypeDB;

class TypeIO {
 private:
  TypeDB& typeDB;

 public:
  explicit TypeIO(TypeDB& config);
  bool load(std::string file);
  bool store(std::string file) const;
};
}  // namespace must

#endif  // LLVM_MUST_SUPPORT_CONFIGIO_H

//
// Created by sebastian on 22.03.18.
//

#ifndef LLVM_MUST_SUPPORT_CONFIGIO_H
#define LLVM_MUST_SUPPORT_CONFIGIO_H

#include <string>
#include <system_error>

namespace typeart {

class TypeDB;

class TypeIO {
 private:
  TypeDB* typeDB;

 public:
  explicit TypeIO(TypeDB* config);
  [[nodiscard]] bool load(const std::string& file, std::error_code& ec);
  [[nodiscard]] bool store(const std::string& file, std::error_code& ec) const;
};

}  // namespace typeart

#endif  // LLVM_MUST_SUPPORT_CONFIGIO_H

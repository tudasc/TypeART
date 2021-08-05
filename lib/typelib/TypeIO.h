//
// Created by sebastian on 22.03.18.
//

#ifndef LLVM_MUST_SUPPORT_CONFIGIO_H
#define LLVM_MUST_SUPPORT_CONFIGIO_H

#include "llvm/Support/ErrorOr.h"

#include <string>

namespace typeart {

class TypeDB;

namespace io {
[[nodiscard]] llvm::ErrorOr<bool> load(TypeDB* db, const std::string& file);
[[nodiscard]] llvm::ErrorOr<bool> store(const TypeDB* db, const std::string& file);
}  // namespace io

}  // namespace typeart

#endif  // LLVM_MUST_SUPPORT_CONFIGIO_H

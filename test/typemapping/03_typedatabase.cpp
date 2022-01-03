// clang-format off
// RUN: %apply %s --manual --object %s.o
// RUN: %clang-cpp %s.o -o %s.exe %types_lib -Wl,-rpath,%typeslib_path
// RUN: %s.exe | %filecheck %s

// RUN: %run %s --manual | %filecheck %s --check-prefix=RUNTIME-LINK

// UNSUPPORTED: sanitizer
// UNSUPPORTED: coverage

// clang-format on

#include "TypeDatabase.h"

#include <cstdio>

int main() {
  auto db = typeart::make_database("types-missing-file.yaml");
  printf("isValid: %i\n", db.first->isValid(256));
  printf("isValid: %i\n", db.first->isValid(1));
  printf("isValid: %i\n", db.first->isValid(258));

  return 0;
}

// CHECK: 0
// CHECK-NEXT: 1
// CHECK-NEXT: 0

// RUNTIME-LINK: 0
// RUNTIME-LINK-NEXT: 1
// RUNTIME-LINK-NEXT: 0
// RUN: %run %s 2>&1 | %filecheck %s

#include "util.h"

struct S1 {
  int x;
  ~S1(){};
};

int main() {
  const auto check = [&](auto* addr, size_t elems) {
    int id_result{-1};
    size_t count_check{0};
    typeart_status status = typeart_get_type(reinterpret_cast<const void*>(addr), &id_result, &count_check);

    if (status == TYPEART_OK) {
      if (count_check != elems) {
        fprintf(stderr, "[Error]: Count not expected: %zu. Expected: %zu.\n", count_check, elems);
      }
    } else {
      fprintf(stderr, "[Check]: Status: %i with #elem %zu.\n", status, elems);
    }
  };

  for (size_t elems = 1; elems < 5; ++elems) {
    // allocates additional sizeof(*size_t*) bytes to store expected count -> array cookie:
    S1* ss = new S1[elems];
    check(ss, elems);
    delete[] ss;
  }

  return 0;
}

// CHECK-NOT: Error
// CHECK-NOT: [Check]: Status: {{[1-9]+}}
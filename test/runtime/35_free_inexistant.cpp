// clang-format off
// RUN: %run %s --manual 2>&1 | %filecheck %s
// clang-format on

#include "../../lib/runtime/CallbackInterface.h"
#include "util.h"

#include <stdio.h>

int main(int argc, char** argv) {
  const int type{6};
  const size_t extent{6};
  const size_t expected_count{extent};

  const auto check = [&](double* addr) {
    int id_result{-1};
    size_t count_check{0};
    typeart_status status = typeart_get_type(reinterpret_cast<const void*>(addr), &id_result, &count_check);

    if (status == TYPEART_OK) {
      if (count_check != expected_count) {
        fprintf(stderr, "[Error]: Count not expected: %zu\n", count_check);
      }
      if (id_result != type) {
        fprintf(stderr, "[Error]: Type not expected: %i\n", id_result);
      }
    } else {
      fprintf(stderr, "[Check]: Status: %i\n", status);
    }
  };

  auto* d = new double[extent];

  // CHECK: [Error]{{.*}}Free on nullptr
  __typeart_free(nullptr);
  // CHECK: [Error]{{.*}}Free on unregistered address
  __typeart_free(reinterpret_cast<const void*>(d));

  // CHECK: [Trace] Alloc 0x{{[0-9a-f]+}} 6 double 8 6
  __typeart_alloc(reinterpret_cast<const void*>(&d[0]), type, extent);
  // CHECK-NOT: [Error]
  // CHECK-NOT: [Check]
  check(&d[0]);

  // CHECK: [Trace] Free 0x{{[0-9a-f]+}} 6 double 8 6
  __typeart_free(reinterpret_cast<const void*>(d));
  // CHECK: [Error]{{.*}}Free on unregistered address
  __typeart_free(reinterpret_cast<const void*>(d));

  delete[] d;

  return 0;
}

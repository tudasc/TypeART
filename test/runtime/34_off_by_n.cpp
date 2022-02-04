// clang-format off
// RUN: %run %s --manual 2>&1 | %filecheck %s
// clang-format on

#include "../../lib/runtime/CallbackInterface.h"
#include "util.h"

#include <stdio.h>

int main(int argc, char** argv) {
  const int type{6};
  const size_t extent{6};
  const size_t expected_count{1};

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

  __typeart_alloc(reinterpret_cast<const void*>(&d[0]), type, 1);
  __typeart_alloc(reinterpret_cast<const void*>(&d[1]), type, 1);

  // CHECK-NOT: [Error]
  check(&d[0]);
  check(&d[1]);
  // CHECK: {{.*}}:Out of bounds for the lookup: (0x{{[0-9a-f]+}} 6 double 8 1 (0x{{[0-9a-f]+}})) #Elements too far: 1
  // CHECK: [Check]: Status: 1
  check(&d[2]);  // one off
  // CHECK: {{.*}}:Out of bounds for the lookup: (0x{{[0-9a-f]+}} 6 double 8 1 (0x{{[0-9a-f]+}})) #Elements too far: 4
  // CHECK: [Check]: Status: 1
  check(&d[5]);  // four off

  // CHECK-NOT: {{.*}}:Out of bounds for the lookup
  // CHECK-NOT: [Error]
  // CHECK: [Check]: Status: 1
  double* p_0 = (&d[0]) - 1;
  check(p_0);  // -1 off

  delete[] d;

  return 0;
}

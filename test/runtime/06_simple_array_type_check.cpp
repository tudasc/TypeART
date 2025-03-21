// RUN: %run %s --compile_flags %dimeta_def 2>&1 | %filecheck %s

#include "../../lib/typelib/TypeInterface.h"
#include "util.h"

#include <stdint.h>
#include <stdlib.h>

template <typename T>
void performTypeChecks(int n, typeart_builtin_type typeId) {
  T* p = (T*)malloc(n * sizeof(T));
  check(p - 1, typeId, 1, 1);                                    // Unknown address
  check(p, typeId, n, 1);                                        // Ok
  check(p + n / 2, typeId, n - n / 2, 1);                        // Ok
  check(p + n - 1, typeId, 1, 1);                                // Ok
  check(p + n, typeId, 1, 1);                                    // Error: Unknown address
  check(((uint8_t*)p) + 1, typeId, n - 1, 1);                    // Fails for sizeof(T) > 1
  check(((uint8_t*)(p + n / 2)) + 1, typeId, n - n / 2 - 1, 1);  // Fails for sizeof(T) > 1
  check(((uint8_t*)p) + 2, typeId, n - 2 / sizeof(T), 1);        // Fails for sizeof(T) > 2
  check(((uint8_t*)p) + 4, typeId, n - 4 / sizeof(T), 1);        // Fails for sizeof(T) > 4
  check(((uint8_t*)p) + 8, typeId, n - 8 / sizeof(T), 1);        // Fails for sizeof(T) > 8
  free(p);
}

int main(int argc, char** argv) {
  const int n = 42;

  // CHECK: [Trace] Alloc 0x{{.*}} {{(int8_t|char)}} 1 42
  // CHECK: Error: Unknown address
  // CHECK: Ok
  // CHECK: Ok
  // CHECK: Ok
  // CHECK: Error: Unknown address
  // CHECK: Ok
  // CHECK: Ok
  // CHECK: Ok
  // CHECK: Ok
  // CHECK: Ok
  // CHECK: [Trace] Free 0x{{.*}}

#if DIMETA == 1
  performTypeChecks<char>(n, TYPEART_CHAR_8);
#else
  performTypeChecks<char>(n, TYPEART_INT_8);
#endif

  // CHECK: [Trace] Alloc 0x{{.*}} short 2 42
  // CHECK: Error: Unknown address
  // CHECK: Ok
  // CHECK: Ok
  // CHECK: Ok
  // CHECK: Error: Unknown address
  // CHECK: Error: Bad alignment
  // CHECK: Error: Bad alignment
  // CHECK: Ok
  // CHECK: Ok
  // CHECK: Ok
  // CHECK: [Trace] Free 0x{{.*}}
  performTypeChecks<short>(n, TYPEART_INT_16);

  // CHECK: [Trace] Alloc 0x{{.*}} int 4 42
  // CHECK: Error: Unknown address
  // CHECK: Ok
  // CHECK: Ok
  // CHECK: Ok
  // CHECK: Error: Unknown address
  // CHECK: Error: Bad alignment
  // CHECK: Error: Bad alignment
  // CHECK: Error: Bad alignment
  // CHECK: Ok
  // CHECK: Ok
  // CHECK: [Trace] Free 0x{{.*}}
  performTypeChecks<int>(n, TYPEART_INT_32);

  // CHECK: [Trace] Alloc 0x{{.*}} long int 8 42
  // CHECK: Error: Unknown address
  // CHECK: Ok
  // CHECK: Ok
  // CHECK: Ok
  // CHECK: Error: Unknown address
  // CHECK: Error: Bad alignment
  // CHECK: Error: Bad alignment
  // CHECK: Error: Bad alignment
  // CHECK: Error: Bad alignment
  // CHECK: Ok
  // CHECK: [Trace] Free 0x{{.*}}
  performTypeChecks<long>(n, TYPEART_INT_64);

  // CHECK: [Trace] Alloc 0x{{.*}} float 4 42
  // CHECK: Error: Unknown address
  // CHECK: Ok
  // CHECK: Ok
  // CHECK: Ok
  // CHECK: Error: Unknown address
  // CHECK: Error: Bad alignment
  // CHECK: Error: Bad alignment
  // CHECK: Error: Bad alignment
  // CHECK: Ok
  // CHECK: Ok
  // CHECK: [Trace] Free 0x{{.*}}
  performTypeChecks<float>(n, TYPEART_FLOAT_32);

  // CHECK: [Trace] Alloc 0x{{.*}} double 8 42
  // CHECK: Error: Unknown address
  // CHECK: Ok
  // CHECK: Ok
  // CHECK: Ok
  // CHECK: Error: Unknown address
  // CHECK: Error: Bad alignment
  // CHECK: Error: Bad alignment
  // CHECK: Error: Bad alignment
  // CHECK: Error: Bad alignment
  // CHECK: Ok
  // CHECK: [Trace] Free 0x{{.*}}
  performTypeChecks<double>(n, TYPEART_FLOAT_64);

  // CHECK: [Trace] Alloc 0x{{.*}} ptr 8 42
  // CHECK: Error: Unknown address
  // CHECK: Ok
  // CHECK: Ok
  // CHECK: Ok
  // CHECK: Error: Unknown address
  // CHECK: Error: Bad alignment
  // CHECK: Error: Bad alignment
  // CHECK: Error: Bad alignment
  // CHECK: Error: Bad alignment
  // CHECK: Ok
  // CHECK: [Trace] Free 0x{{.*}}
  performTypeChecks<int*>(n, TYPEART_POINTER);

  return 0;
}

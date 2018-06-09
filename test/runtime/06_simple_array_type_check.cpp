// RUN: %scriptpath/applyAndRun.sh %s %pluginpath "-must-alloca" %rtpath | FileCheck %s

#include <stdlib.h>
#include <stdint.h>
#include "util.h"
#include "../../typelib/TypeInterface.h"

template<typename T>
void performTypeChecks(int n, must_builtin_type typeId) {
  T* p = (T*) malloc(n * sizeof(T));
  check(p - 1, typeId, 1, 1);  // Unknown address
  check(p, typeId, n, 1);  // Ok
  check(p + n/2, typeId, n - n/2, 1); // Ok
  check(p + n-1, typeId, 1, 1); // Ok
  check(p + n, typeId, 1, 1); // Error: Unknown address
  check(((uint8_t*)p) + 1, typeId, n-1, 1); // Fails for sizeof(T) > 1
  check(((uint8_t*) (p + n/2)) + 1, typeId, n - n/2 - 1, 1); // Fails for sizeof(T) > 1
  check(((uint8_t*)p) + 2, typeId, n-2/sizeof(T), 1); // Fails for sizeof(T) > 2
  check(((uint8_t*)p) + 4, typeId, n-4/sizeof(T), 1); // Fails for sizeof(T) > 4
  check(((uint8_t*)p) + 8, typeId, n-8/sizeof(T), 1); // Fails for sizeof(T) > 8
  free(p);
}


int main(int argc, char** argv) {

  const int n = 42;

  // CHECK: Alloc    0x{{.*}}    char    1   42
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
  // CHECK: Free 0x{{.*}}
  performTypeChecks<char>(n, C_CHAR);

  // CHECK: Alloc    0x{{.*}}    short    2   42
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
  // CHECK: Free 0x{{.*}}
  performTypeChecks<short>(n, C_SHORT);

  // CHECK: Alloc    0x{{.*}}    int    4   42
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
  // CHECK: Free 0x{{.*}}
  performTypeChecks<int>(n, C_INT);

  // CHECK: Alloc    0x{{.*}}    long    8   42
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
  // CHECK: Free 0x{{.*}}
  performTypeChecks<long>(n, C_LONG);

  // CHECK: Alloc    0x{{.*}}    float    4   42
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
  // CHECK: Free 0x{{.*}}
  performTypeChecks<float>(n, C_FLOAT);

  // CHECK: Alloc    0x{{.*}}    double    8   42
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
  // CHECK: Free 0x{{.*}}
  performTypeChecks<double>(n, C_DOUBLE);

  // CHECK: Alloc    0x{{.*}}    unknown    8   42
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
  // CHECK: Free 0x{{.*}}
  performTypeChecks<int*>(n, UNKNOWN);

  return 0;
}

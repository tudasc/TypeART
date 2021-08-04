// RUN: TYPEART_TYPE_FILE=%p%{pathsep}%t %run %s 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL-FILE
// XFAIL: *

#include <stdio.h>
#include <stdlib.h>

struct Datastruct {
  int start;
  double middle;
  float end;
};

int main(int argc, char** argv) {
  struct Datastruct data = {0};
  return data.start;
}

// Nonexistant (using environment var) file aborts runtime:
// CHECK-FAIL-FILE: [Fatal]{{.*}}Failed to load recorded types
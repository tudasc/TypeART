// RUN: TYPEART_TYPE_FILE=%p%{pathsep}%t %run %s 2>&1 | %filecheck %s --check-prefix=CHECK-FAIL-FILE
// RUN: TA_TYPE_FILE=%p%{pathsep}%t %run %s 2>&1 | %filecheck %s --check-prefix=CHECK-FAIL-FILE-DEP

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

// Nonexistent (using environment var) file aborts runtime:
// CHECK-FAIL-FILE: [Fatal]{{.*}}Failed to load recorded types

// CHECK-FAIL-FILE-DEP: [WARNING]{{.*}}Use of deprecated env var
// CHECK-FAIL-FILE-DEP: [Fatal]{{.*}}Failed to load recorded types
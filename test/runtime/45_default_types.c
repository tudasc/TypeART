// RUN: rm types.yaml
// RUN: %run %s 2>&1 | FileCheck %s

// XFAIL: *


int main(int argc, char** argv) {
  return 0;
}

// CHECK: [FATAL]{{.*}}No type file with default name
// RUN: %run %s 2>&1 | %filecheck %s

#include "../../lib/runtime/RuntimeInterface.h"

#include <stdbool.h>
#include <stdio.h>

struct Datastruct {
  int start;
  double middle;
  float end;
};

int main(int argc, char** argv) {
  printf("Version: %s\n", typeart_get_project_version());
  printf("Revision: %s\n", typeart_get_git_revision());
  printf("LLVM version: %s\n", typeart_get_llvm_version());

  return 0;
}

// CHECK: Version: {{[1-9]+.[0-9]+.?[0-9]?[0-9]?}}
// CHECK: Revision: {{[a-zA-Z0-9-]+}}
// CHECK: LLVM version: {{[0-9]+.[0-9]+}}
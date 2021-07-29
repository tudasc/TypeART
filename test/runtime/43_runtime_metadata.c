// RUN: %run %s 2>&1 | FileCheck %s

#include "../../lib/runtime/RuntimeInterface.h"

#include <stdio.h>
#include <stdbool.h>

struct Datastruct {
  int start;
  double middle;
  float end;
};

int main(int argc, char** argv) {
  printf("Version: %s\n", typeart_get_project_version());
  printf("Revision: %s\n",typeart_get_git_revision());

  return 0;
}

// CHECK: Version: {{[1-9]+.[1-9]+.?[1-9]?[1-9]?}}
// CHECK: Revision: {{[a-zA-Z0-9-]+}}
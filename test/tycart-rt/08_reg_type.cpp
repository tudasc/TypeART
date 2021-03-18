// RUN: %scriptpath/applyAndRun-tycart.sh %s %pluginpath 2>&1 | FileCheck %s

#include "support.h"

struct S1 {
  int x;
  virtual ~S1() = default;
};


int main() {
  S1 s;
  TY_register_type(S1);  
  return 0;
}

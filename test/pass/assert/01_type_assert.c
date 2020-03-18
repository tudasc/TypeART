// clang-format off
// RUN: clang -I../../../runtime -I../../../typelib -S -emit-llvm %s -o - | opt -load %pluginpath/analysis/meminstfinderpass.so -load %pluginpath/%pluginname %pluginargs -S 2>&1 | FileCheck %s
// clang-format on

#include "RuntimeInterface.h"

#define make_test(type) \
{ \
  type x; \
  ASSERT_TYPE(&x, type); \
} \

struct S {
  int x;
};

void test() {
  // CHECK: Resolved assert (id=0)
  make_test(char);
  // CHECK: Resolved assert (id=1)
  make_test(short);
  // CHECK: Resolved assert (id=2)
  make_test(int);
  // CHECK: Resolved assert (id=3)
  make_test(long);
  // CHECK: Resolved assert (id=5)
  make_test(float);
  // CHECK: Resolved assert (id=6)
  make_test(double);
  // CHECK: Resolved assert (id=10)
  make_test(char*);
  // CHECK: Resolved assert (id=256)
  make_test(struct S);
}

// CHECK: Malloc{{[ ]*}}:{{[ ]*}}0
// CHECK: Free{{[ ]*}}:{{[ ]*}}0
// CHECK: Alloca{{[ ]*}}:{{[ ]*}}1

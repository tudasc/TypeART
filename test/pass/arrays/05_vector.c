// clang-format off
// RUN: clang -S -emit-llvm %s -o - | opt -load %pluginpath/analysis/meminstfinderpass.so -load %pluginpath/%pluginname %pluginargs -S 2>&1 | FileCheck %s
// clang-format on

typedef float float2 __attribute__((ext_vector_type(2)));

void test() {
  float2 vf = (float2)(1.0f, 2.0f);
}

// CHECK-NOT Type is not supported: <2 x float>
// CHECK: alloca <2 x float>, align 8
// CHECK: Malloc{{[ ]*}}:{{[ ]*}}0
// CHECK: Free{{[ ]*}}:{{[ ]*}}0
// CHECK: Alloca{{[ ]*}}:{{[ ]*}}1

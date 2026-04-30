// clang-format off
// RUN: %hip-cpp-to-llvm %s | TYPEART_GPU=true %apply-typeart -typeart-type-serialization=inline -S 2>&1 | %filecheck %s --check-prefix INLINE
// RUN: %hip-cpp-to-llvm %s | TYPEART_GPU=true %apply-typeart -typeart-type-serialization=hybrid -S 2>&1 | %filecheck %s --check-prefix HYBRID
// RUN: %hip-cpp-to-llvm %s | TYPEART_GPU=true %apply-typeart -typeart-type-serialization=file -S 2>&1 | %filecheck %s --check-prefix FILE
// clang-format on

// REQUIRES: hip && !llvm-14

// INLINE: %struct._typeart_struct_layout_t = type { i32, i32, ptr }
// INLINE: call void @__typeart_alloc_mty_gpu(ptr %{{[0-9a-z]+}}, ptr @_typeart_{{.*}}, i64 {{.*}})
// INLINE: call void @__typeart_register_type(ptr @_typeart_{{.*}})

// HYBRID: call void @__typeart_alloc_gpu(ptr %{{[0-9a-z]+}}, i32 {{[0-9]+}}, i64 {{.*}})
// HYBRID: call void @__typeart_alloc_mty_gpu(ptr %{{[0-9a-z]+}}, ptr @_typeart_{{.*}}, i64 {{.*}})
// HYBRID: call void @__typeart_register_type(ptr @_typeart_{{.*}})

// FILE: call void @__typeart_alloc_gpu(ptr %{{[0-9a-z]+}}, i32 {{[0-9]+}}, i64 {{.*}})

#include <hip/hip_runtime.h>
typedef struct MyData {
  int a;
  double b;
} MyData;

int main() {
  const int N = 20;
  float* d_x;
  MyData* d_s;

  hipMalloc((void**)&d_x, N * sizeof(float));
  hipMalloc((void**)&d_s, N * sizeof(MyData));

  return 0;
}

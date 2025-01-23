// RUN: %remove %tu_yaml && %c-to-llvm -mavx %s | %opt -O2 -S | %apply-typeart --typeart-stack=true -S 2>&1 | %filecheck %s

#include <immintrin.h>

__m256 vec_result;

void foo(float a[8]) {
  float b[8]   = {8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
  __m256 vec_a = _mm256_loadu_ps(a);
  __m256 vec_b = _mm256_loadu_ps(b);
  vec_result   = _mm256_add_ps(vec_a, vec_b);
}

// CHECK: @__typeart_alloc_global({{.*}}@vec_result{{.*}}, i32 256, i64 1)

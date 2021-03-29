// clang-format off
// RUN: OMP_NUM_THREADS=3 %run %s -o -O1 --omp 2>&1 | FileCheck %s
// REQUIRES: openmp && softcounter
// clang-format on

#include "../../lib/runtime/CallbackInterface.h"

#include <stdlib.h>
#include <vector>
#include <algorithm>

template <typename S, typename E>
void repeat_alloc(S s, E e) {
  std::for_each(s, e, [&](auto& elem) { __typeart_alloc(reinterpret_cast<const void*>(elem), 6, 20); });
}

template <typename S, typename E>
void repeat_dealloc(S s, E e) {
  std::for_each(s, e, [&](auto& elem) { __typeart_free(reinterpret_cast<const void*>(elem)); });
}

std::vector<int> unique_rand(const unsigned size) {
  std::srand(42);
  std::vector<int> vec(size);
  std::generate(vec.begin(), vec.end(), std::rand);

  sort(vec.begin(), vec.end());
  vec.erase(unique(vec.begin(), vec.end()), vec.end());

  return vec;
}

int main(int argc, char** argv) {
  constexpr unsigned size = 300;
  auto vec                = unique_rand(size);
  auto beg                = std::begin(vec);
  auto h1                 = beg + (size / 3);
  auto h2                 = h1 + (size / 3);
  auto e                  = std::end(vec);

#pragma omp parallel sections num_threads(3)
  {
#pragma omp section
    { repeat_alloc(beg, h1); }
#pragma omp section
    { repeat_alloc(h2, e); }
#pragma omp section
    { repeat_alloc(h1, h2); }
  }

#pragma omp parallel sections num_threads(3)
  {
#pragma omp section
    { repeat_dealloc(beg, h1); }
#pragma omp section
    { repeat_dealloc(h2, e); }
#pragma omp section
    { repeat_dealloc(h1, h2); }
  }

  // CHECK-NOT: [Error]

  // CHECK: Allocation type detail (heap, stack, global)
  // CHECK: 6   : 300 ,     0 ,    0 , double

  // CHECK: Free allocation type detail (heap, stack)
  // CHECK: 6   : 300 ,     0 , double
  return 0;
}

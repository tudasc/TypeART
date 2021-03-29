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

std::vector<int> unique_rand(const unsigned size) {
  std::srand(42);
  std::vector<int> vec(size);
  std::generate(vec.begin(), vec.end(), std::rand);

  sort(vec.begin(), vec.end());
  vec.erase(unique(vec.begin(), vec.end()), vec.end());

  return vec;
}

int main(int argc, char** argv) {
  constexpr unsigned size = 100;
  auto vec                = unique_rand(size);
  auto beg                = std::begin(vec);
  auto e                  = std::end(vec);

#pragma omp parallel sections num_threads(3)
  {
#pragma omp section
    { repeat_alloc(beg, e); }
#pragma omp section
    { repeat_alloc(beg, e); }
#pragma omp section
    { repeat_alloc(beg, e); }
  }

  // CHECK-NOT: [Error]

  // 3 Threads, using the same 100 pointers, i.e., 200 are overriden:
  // CHECK: Alloc Stats from softcounters
  // CHECK: Addresses re-used          :  200

  // CHECK: Allocation type detail (heap, stack, global)
  // CHECK: 6   : 300 ,     0 ,    0 , double
  return 0;
}

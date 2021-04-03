// clang-format off
// RUN: %run %s -o -O1 --thread --manual 2>&1 | FileCheck %s
// REQUIRES: thread && softcounter
// clang-format on

#include "../../lib/runtime/CallbackInterface.h"
#include "util.h"

#include <vector>
#include <algorithm>
#include <atomic>
#include <thread>
#include <chrono>
#include <random>

std::atomic_bool stop{false};

template <typename S, typename E>
void repeat_alloc(S s, E e) {
  std::for_each(s, e, [&](auto elem) { __typeart_alloc(reinterpret_cast<const void*>(elem), int{6}, size_t{20}); });
}

template <typename S, typename E>
void repeat_alloc_free_v2(S s, E e) {
  using namespace std::chrono_literals;
  std::for_each(s, e, [&](auto elem) {
    __typeart_alloc(reinterpret_cast<const void*>(elem), int{7}, size_t{10});
    std::this_thread::sleep_for(1ms);
    __typeart_free(reinterpret_cast<const void*>(elem));
  });
}

template <typename S, typename E>
void repeat_type_check(S s, E e) {
  do {
    std::for_each(s, e, [&](auto addr) {
      int id_result{-1};
      size_t count_check{0};
      typeart_status status = typeart_get_type(reinterpret_cast<const void*>(addr), &id_result, &count_check);
      if (status == TA_OK) {
        if (count_check != size_t{20}) {
          fprintf(stderr, "[Error]: Length mismatch of %i (%#02x) is: type=%i count=%zu\n", addr, addr, id_result,
                  count_check);
        }
        if (id_result != int{6}) {
          fprintf(stderr, "[Error]: Type mismatch of %i (%#02x) is: type=%i count=%zu\n", addr, addr, id_result,
                  count_check);
        }
      }
    });
  } while (!stop);
}

std::vector<int> unique_rand(const unsigned size) {
  std::vector<int> vec(size);
  std::iota(vec.begin(), vec.end(), 1);

  std::mt19937 g(42);

  std::shuffle(vec.begin(), vec.end(), g);

  return vec;
}

int main(int argc, char** argv) {
  constexpr unsigned size = 200;
  auto vec                = unique_rand(size);
  auto beg                = std::begin(vec);
  auto h                  = beg + (size / 2);
  auto e                  = std::end(vec);

  std::thread malloc_1(repeat_alloc<decltype(beg), decltype(h)>, beg, h);
  std::thread malloc_2(repeat_alloc_free_v2<decltype(h), decltype(e)>, h, e);

  std::thread check_1(repeat_type_check<decltype(beg), decltype(e)>, beg, h);

  malloc_1.join();
  malloc_2.join();

  stop = true;

  check_1.join();

  // CHECK-NOT: [Error]

  // CHECK: Alloc Stats from softcounters
  // CHECK: Distinct Addresses checked :   100 ,    - ,    -

  // CHECK: Allocation type detail (heap, stack, global)
  // CHECK: 6   : 100 ,     0 ,    0 , double
  // CHECK: 7   : 100 ,      0 ,    0 , float128

  // We free only 3/4 of allocations
  // CHECK: Free allocation type detail (heap, stack)
  // CHECK: 7   : 100 ,     0 , float128
  return 0;
}

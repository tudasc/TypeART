// clang-format off
// RUN: %run %s -o -O1 --thread --manual 2>&1 | %filecheck %s --check-prefix=CHECK-TSAN
// RUN: %run %s -o -O1 --thread --manual 2>&1 | %filecheck %s
// REQUIRES: thread && softcounter
// clang-format on

#include "../../lib/runtime/CallbackInterface.h"
#include "util.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <random>
#include <thread>
#include <vector>

std::atomic_bool stop{false};
const size_t extent{1};

template <typename S, typename E>
void repeat_alloc(S s, E e) {
  std::for_each(s, e, [&](auto elem) { __typeart_alloc(reinterpret_cast<const void*>(elem), int{6}, extent); });
}

template <typename S, typename E>
void repeat_alloc_free_v2(S s, E e) {
  using namespace std::chrono_literals;
  std::for_each(s, e, [&](auto elem) {
    __typeart_alloc(reinterpret_cast<const void*>(elem), int{7}, extent);
    // std::this_thread::sleep_for(1ms);
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
      if (status == TYPEART_OK) {
        if (count_check != extent) {
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
  unsigned cnt{1};
  std::generate(vec.begin(), vec.end(), [&cnt]() {
    auto current = cnt;
    cnt += extent * sizeof(double);
    return current;
  });
  std::mt19937 g(42);
  std::shuffle(vec.begin(), vec.end(), g);
  return vec;
}

int main(int argc, char** argv) {
  constexpr unsigned size = 100;
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

  // CHECK-TSAN-NOT: ThreadSanitizer
  // CHECK-NOT: Error
  return 0;
}

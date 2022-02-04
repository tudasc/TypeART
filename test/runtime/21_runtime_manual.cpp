// clang-format off
// RUN: clang++ -std=c++17 -I%S/../../ -I%S/../../lib/typelib -I%S/../../lib  %s -o %s.exe
// RUN: %s.exe 2>&1 | %filecheck %s
// clang-format on
// FIXME this test doesn't add to the coverage data.

#define ENABLE_SOFTCOUNTER 1
#include "lib/runtime/AccessCounter.h"

#include <algorithm>
#include <iostream>

using namespace typeart;

#define o_(getter) std::cerr << recorder.getter << '\n'

std::vector<std::pair<int, softcounter::Counter>> sorted_v(const std::unordered_map<int, softcounter::Counter>& map) {
  std::vector<std::pair<int, softcounter::Counter>> sorted_elements(map.begin(), map.end());
  std::sort(sorted_elements.begin(), sorted_elements.end());
  return sorted_elements;
}

std::vector<MemAddr> sorted_v(const std::unordered_set<MemAddr>& set) {
  std::vector<MemAddr> sorted_elements(set.begin(), set.end());
  std::sort(sorted_elements.begin(), sorted_elements.end());
  return sorted_elements;
}

void test_heap(softcounter::AccessRecorder& recorder) {
  recorder.incHeapAlloc(10, 1);
  recorder.incHeapAlloc(10, 1);

  // CHECK: 2
  o_(getCurHeapAllocs());
  // CHECK: 2
  o_(getMaxHeapAllocs());

  auto hallocs = sorted_v(recorder.getHeapAlloc());
  // CHECK: 1
  std::cerr << hallocs.size() << '\n';
  // CHECK: 10 2
  for (const auto& [id, count] : hallocs) {
    std::cerr << id << " " << count << '\n';
  }

  recorder.decHeapAlloc();
  recorder.decHeapAlloc();
  // CHECK: 0
  o_(getCurHeapAllocs());
  // CHECK: 2
  o_(getMaxHeapAllocs());

  recorder.decHeapAlloc();
  recorder.decHeapAlloc();
  // CHECK: -2
  o_(getCurHeapAllocs());
  // CHECK: 2
  o_(getMaxHeapAllocs());
}

void test_stack(softcounter::AccessRecorder& recorder) {
  recorder.incStackAlloc(0, 1);
  // CHECK: 1
  o_(getStackAllocs());
  // CHECK: 0
  o_(getMaxStackAllocs());

  recorder.incStackAlloc(1, 1);
  // CHECK: 2
  o_(getStackAllocs());
  // CHECK: 0
  o_(getMaxStackAllocs());

  recorder.decStackAlloc(2);
  // CHECK: 2
  o_(getStackAllocs());
  // CHECK: 2
  o_(getMaxStackAllocs());

  auto sallocs = sorted_v(recorder.getStackAlloc());
  // CHECK: 2
  std::cerr << sallocs.size() << '\n';
  // CHECK: 0 1
  // CHECK: 1 1
  for (const auto& [id, count] : sallocs) {
    std::cerr << id << " " << count << '\n';
  }

  auto de_sallocs = sorted_v(recorder.getStackFree());
  // CHECK: 0
  std::cerr << de_sallocs.size() << '\n';

  recorder.incStackFree(0, 1);
  recorder.incStackFree(1, 1);
  de_sallocs = sorted_v(recorder.getStackFree());
  // CHECK: 2
  std::cerr << de_sallocs.size() << '\n';
  // CHECK: 0 1
  // CHECK: 1 1
  for (const auto& [id, count] : de_sallocs) {
    std::cerr << id << " " << count << '\n';
  }

  recorder.incStackAlloc(6, 1);
  recorder.incStackFree(6, 1);
  de_sallocs = sorted_v(recorder.getStackFree());
  // CHECK: 3
  std::cerr << de_sallocs.size() << '\n';
  // CHECK: 0 1
  // CHECK: 1 1
  // CHECK: 6 1
  for (const auto& [id, count] : de_sallocs) {
    std::cerr << id << " " << count << '\n';
  }
}

void test_global(softcounter::AccessRecorder& recorder) {
  recorder.incGlobalAlloc(6, 1);
  // CHECK: 1
  o_(getGlobalAllocs());

  const auto& alloc = recorder.getGlobalAlloc();
  // CHECK: 1
  std::cerr << alloc.size() << '\n';
  // CHECK: 6 1
  for (const auto& [id, count] : alloc) {
    std::cerr << id << " " << count << '\n';
  }
}

int main() {
  softcounter::AccessRecorder recorder;

  test_heap(recorder);
  test_stack(recorder);
  test_global(recorder);

  recorder.incUDefTypes(2);
  // CHECK: 2
  o_(getNumUDefTypes());

  void* a1 = (void*)0x1;
  void* a2 = (void*)0x2;
  void* a3 = (void*)0x3;
  recorder.incAddrMissing(a1);
  // CHECK: 1
  o_(getAddrMissing());
  recorder.incAddrMissing(a2);
  recorder.incAddrMissing(a3);
  // CHECK: 3
  o_(getAddrMissing());
  recorder.incAddrMissing(a3);
  // CHECK: 4
  o_(getAddrMissing());

  auto mset = sorted_v(recorder.getMissing());
  // CHECK: 3
  std::cerr << mset.size() << '\n';
  // CHECK: 0x1
  // CHECK: 0x2
  // CHECK: 0x3
  for (auto& a : mset) {
    std::cerr << a << '\n';
  }

  recorder.incUsedInRequest(a1);
  recorder.incUsedInRequest(a3);
  // CHECK: 2
  o_(getAddrChecked());

  auto cset = sorted_v(recorder.getSeen());
  // CHECK: 2
  std::cerr << cset.size() << '\n';
  // CHECK: 0x1
  // CHECK: 0x3
  for (auto& a : cset) {
    std::cerr << a << '\n';
  }
  return 0;
}

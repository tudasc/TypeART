//
// Created by ahueck on 12.10.20.
//

#ifndef TYPEART_ACCESSCOUNTER_H
#define TYPEART_ACCESSCOUNTER_H

#include "RuntimeData.h"
#include "RuntimeInterface.h"

#include <atomic>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace typeart {

namespace softcounter {

using Counter       = long long int;
using AtomicCounter = std::atomic<Counter>;

class AccessRecorder {
 public:
  using TypeCountMap = std::unordered_map<int, Counter>;
  ~AccessRecorder()  = default;

  inline void incHeapAlloc(int typeId, size_t count) {
    ++curHeapAllocs;

    std::lock_guard<std::mutex> guard(recorderMutex);

    // Always check here for max
    // A program without free would otherwise never update maxHeap (see test 20_softcounter_max)
    if (curHeapAllocs > maxHeapAllocs) {
      maxHeapAllocs = curHeapAllocs.load();
    }

    ++heapAllocs;
    if (count > 1) {
      ++heapArray;
    }
    ++heapAlloc[typeId];
  }

  inline void incStackAlloc(int typeId, size_t count) {
    ++curStackAllocs;
    ++stackAllocs;
    if (count > 1) {
      ++stackArray;
    }

    std::lock_guard<std::mutex> guard(recorderMutex);
    ++stackAlloc[typeId];
  }

  inline void incGlobalAlloc(int typeId, size_t count) {
    ++globalAllocs;
    if (count > 1) {
      ++globalArray;
    }

    std::lock_guard<std::mutex> guard(recorderMutex);

    ++globalAlloc[typeId];
  }

  inline void incStackFree(int typeId, size_t count) {
    ++stackAllocsFree;
    if (count > 1) {
      ++stackArrayFree;
    }

    std::lock_guard<std::mutex> guard(recorderMutex);

    ++stackFree[typeId];
  }

  inline void incHeapFree(int typeId, size_t count) {
    ++heapAllocsFree;
    if (count > 1) {
      ++heapArrayFree;
    }

    std::lock_guard<std::mutex> guard(recorderMutex);

    ++heapFree[typeId];
  }

  inline void decHeapAlloc() {
    // Removed, since we already increment maxHeapAllocs just in time:
    //    if (curHeapAllocs > maxHeapAllocs) {
    //      maxHeapAllocs = curHeapAllocs;
    //    }
    --curHeapAllocs;
  }

  inline void decStackAlloc(size_t amount) {
    std::lock_guard<std::mutex> guard(recorderMutex);
    if (curStackAllocs > maxStackAllocs) {
      maxStackAllocs = curStackAllocs.load();
    }
    curStackAllocs -= amount;
  }

  inline void incUsedInRequest(MemAddr addr) {
    std::lock_guard<std::mutex> guard(recorderMutex);
    ++addrChecked;
    seen.insert(addr);
  }

  inline void incAddrReuse() {
    ++addrReuses;
  }

  inline void incAddrMissing(MemAddr addr) {
    std::lock_guard<std::mutex> guard(recorderMutex);
    ++addrMissing;
    missing.insert(addr);
  }

  inline void incNullAddr() {
    ++nullAlloc;
  }

  inline void incZeroLengthAddr() {
    ++zeroAlloc;
  }

  inline void incZeroLengthAndNullAddr() {
    ++nullAndZeroAlloc;
  }

  inline void incUDefTypes(size_t count) {
    numUDefTypes += count;
  }

  inline void incOmpContextStack() {
    ++omp_stack;
  }

  inline void incOmpContextHeap() {
    ++omp_heap;
  }

  inline void incOmpContextFree() {
    ++omp_heap_free;
  }

  Counter getHeapAllocs() const {
    return heapAllocs;
  }
  Counter getStackAllocs() const {
    return stackAllocs;
  }
  Counter getGlobalAllocs() const {
    return globalAllocs;
  }
  Counter getMaxHeapAllocs() const {
    return maxHeapAllocs;
  }
  Counter getMaxStackAllocs() const {
    return maxStackAllocs;
  }
  Counter getCurHeapAllocs() const {
    return curHeapAllocs;
  }
  Counter getCurStackAllocs() const {
    return curStackAllocs;
  }
  Counter getAddrReuses() const {
    return addrReuses;
  }
  Counter getAddrMissing() const {
    return addrMissing;
  }
  Counter getAddrChecked() const {
    return addrChecked;
  }
  Counter getStackArray() const {
    return stackArray;
  }
  Counter getHeapArray() const {
    return heapArray;
  }
  Counter getGlobalArray() const {
    return globalArray;
  }
  Counter getStackAllocsFree() const {
    return stackAllocsFree;
  }
  Counter getStackArrayFree() const {
    return stackArrayFree;
  }
  Counter getHeapAllocsFree() const {
    return heapAllocsFree;
  }
  Counter getHeapArrayFree() const {
    return heapArrayFree;
  }
  Counter getNullAlloc() const {
    return nullAlloc;
  }
  Counter getZeroAlloc() const {
    return zeroAlloc;
  }
  Counter getNullAndZeroAlloc() const {
    return nullAndZeroAlloc;
  }
  Counter getNumUDefTypes() const {
    return numUDefTypes;
  }
  Counter getOmpHeapCalls() const {
    return omp_heap;
  }
  Counter getOmpFreeCalls() const {
    return omp_heap_free;
  }
  Counter getOmpStackCalls() const {
    return omp_stack;
  }
  const std::unordered_set<MemAddr>& getMissing() const {
    return missing;
  }
  const std::unordered_set<MemAddr>& getSeen() const {
    return seen;
  }
  const TypeCountMap& getStackAlloc() const {
    return stackAlloc;
  }
  const TypeCountMap& getHeapAlloc() const {
    return heapAlloc;
  }
  const TypeCountMap& getGlobalAlloc() const {
    return globalAlloc;
  }
  const TypeCountMap& getStackFree() const {
    return stackFree;
  }
  const TypeCountMap& getHeapFree() const {
    return heapFree;
  }

 private:
  AtomicCounter heapAllocs       = 0;
  AtomicCounter stackAllocs      = 0;
  AtomicCounter globalAllocs     = 0;
  AtomicCounter maxHeapAllocs    = 0;
  AtomicCounter maxStackAllocs   = 0;
  AtomicCounter curHeapAllocs    = 0;
  AtomicCounter curStackAllocs   = 0;
  AtomicCounter addrReuses       = 0;
  AtomicCounter addrMissing      = 0;
  AtomicCounter addrChecked      = 0;
  AtomicCounter stackArray       = 0;
  AtomicCounter heapArray        = 0;
  AtomicCounter globalArray      = 0;
  AtomicCounter stackAllocsFree  = 0;
  AtomicCounter stackArrayFree   = 0;
  AtomicCounter heapAllocsFree   = 0;
  AtomicCounter heapArrayFree    = 0;
  AtomicCounter nullAlloc        = 0;
  AtomicCounter zeroAlloc        = 0;
  AtomicCounter nullAndZeroAlloc = 0;
  AtomicCounter numUDefTypes     = 0;
  AtomicCounter omp_stack        = 0;
  AtomicCounter omp_heap         = 0;
  AtomicCounter omp_heap_free    = 0;

  std::unordered_set<MemAddr> missing;
  std::unordered_set<MemAddr> seen;
  TypeCountMap stackAlloc;
  TypeCountMap heapAlloc;
  TypeCountMap globalAlloc;
  TypeCountMap stackFree;
  TypeCountMap heapFree;

  std::mutex recorderMutex;
};

/**
 * Used for no-operations in counter methods when not using softcounters.
 */
class NoneRecorder {
 public:
  [[maybe_unused]] inline void incHeapAlloc(int, size_t) {
  }
  [[maybe_unused]] inline void incStackAlloc(int, size_t) {
  }
  [[maybe_unused]] inline void incGlobalAlloc(int, size_t) {
  }
  [[maybe_unused]] inline void incUsedInRequest(MemAddr) {
  }
  [[maybe_unused]] inline void decHeapAlloc() {
  }
  [[maybe_unused]] inline void decStackAlloc(size_t) {
  }
  [[maybe_unused]] inline void incAddrReuse() {
  }
  [[maybe_unused]] inline void incAddrMissing(MemAddr) {
  }
  [[maybe_unused]] inline void incStackFree(int, size_t) {
  }
  [[maybe_unused]] inline void incHeapFree(int, size_t) {
  }
  [[maybe_unused]] inline void incNullAddr() {
  }
  [[maybe_unused]] inline void incZeroLengthAddr() {
  }
  [[maybe_unused]] inline void incZeroLengthAndNullAddr() {
  }
  [[maybe_unused]] inline void incUDefTypes(size_t count) {
  }
  [[maybe_unused]] inline void incOmpContextStack() {
  }
  [[maybe_unused]] inline void incOmpContextHeap() {
  }
  [[maybe_unused]] inline void incOmpContextFree() {
  }
};

}  // namespace softcounter

#if ENABLE_SOFTCOUNTER == 1
using Recorder = softcounter::AccessRecorder;
#else
using Recorder = softcounter::NoneRecorder;
#endif

}  // namespace typeart

#endif  // TYPEART_ACCESSCOUNTER_H

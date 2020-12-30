//
// Created by ahueck on 12.10.20.
//

#ifndef TYPEART_ACCESSCOUNTER_H
#define TYPEART_ACCESSCOUNTER_H

#include "RuntimeData.h"
#include "RuntimeInterface.h"

#include <map>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace typeart {

namespace softcounter {

using Counter = long long int;

class AccessRecorder {
 public:
  using TypeCountMap = std::unordered_map<int, Counter>;
  ~AccessRecorder()  = default;

  inline void incHeapAlloc(int typeId, size_t count) {
    ++curHeapAllocs;

    // Always check here for max
    // A program without free would otherwise never update maxHeap (see test 20_softcounter_max)
    if (curHeapAllocs > maxHeapAllocs) {
      maxHeapAllocs = curHeapAllocs;
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
    ++stackAlloc[typeId];
  }

  inline void incGlobalAlloc(int typeId, size_t count) {
    ++globalAllocs;
    if (count > 1) {
      ++globalArray;
    }
    ++globalAlloc[typeId];
  }

  inline void incStackFree(int typeId, size_t count) {
    ++stackAllocsFree;
    if (count > 1) {
      ++stackArrayFree;
    }
    ++stackFree[typeId];
  }

  inline void incHeapFree(int typeId, size_t count) {
    ++heapAllocsFree;
    if (count > 1) {
      ++heapArrayFree;
    }
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
    if (curStackAllocs > maxStackAllocs) {
      maxStackAllocs = curStackAllocs;
    }
    curStackAllocs -= amount;
  }

  inline void incUsedInRequest(MemAddr addr) {
    ++addrChecked;
    seen.insert(addr);
  }

  inline void incAddrReuse() {
    ++addrReuses;
  }

  inline void incAddrMissing(MemAddr addr) {
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

  [[nodiscard]] static AccessRecorder& get() {
    static AccessRecorder instance;
    return instance;
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
  AccessRecorder()                       = default;
  AccessRecorder(AccessRecorder& other)  = default;
  AccessRecorder(AccessRecorder&& other) = default;

  Counter heapAllocs       = 0;
  Counter stackAllocs      = 0;
  Counter globalAllocs     = 0;
  Counter maxHeapAllocs    = 0;
  Counter maxStackAllocs   = 0;
  Counter curHeapAllocs    = 0;
  Counter curStackAllocs   = 0;
  Counter addrReuses       = 0;
  Counter addrMissing      = 0;
  Counter addrChecked      = 0;
  Counter stackArray       = 0;
  Counter heapArray        = 0;
  Counter globalArray      = 0;
  Counter stackAllocsFree  = 0;
  Counter stackArrayFree   = 0;
  Counter heapAllocsFree   = 0;
  Counter heapArrayFree    = 0;
  Counter nullAlloc        = 0;
  Counter zeroAlloc        = 0;
  Counter nullAndZeroAlloc = 0;
  Counter numUDefTypes     = 0;
  std::unordered_set<MemAddr> missing;
  std::unordered_set<MemAddr> seen;
  TypeCountMap stackAlloc;
  TypeCountMap heapAlloc;
  TypeCountMap globalAlloc;
  TypeCountMap stackFree;
  TypeCountMap heapFree;
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

  static NoneRecorder& get() {
    static NoneRecorder instance;
    return instance;
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

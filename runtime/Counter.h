/*
 * Counter.h
 *
 *  Created on: Mar 17, 2020
 *      Author: ahueck
 */

#ifndef RUNTIME_COUNTER_H_
#define RUNTIME_COUNTER_H_

#include "Logger.h"
#include "RuntimeInterface.h"

#include <llvm/Support/raw_ostream.h>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace typeart {
namespace softcounter {
/**
 * Very basic implementation of some couting infrastructure.
 * This implementation counts:
 * - the number of objects hold maximally in the datastructures for stack and heap.
 * - the total number of tracked allocations (counting multiple insertions of the same address as multiple tracked
 * values) for both stack and heap.
 * - the number of distinct addresses queried for information
 * - the number of addresses re-used (according to our type map)
 * In addition it estimates (lower-bound) the consumed memory for tracking the type information.
 *
 * It prints the information during object de-construction.
 */
class AccessRecorder {
 public:
  using type_count_map = std::unordered_map<int, long long>;
  ~AccessRecorder() {
    printStats();
  }

  inline void incHeapAlloc(int typeId, size_t count) {
    ++curHeapAllocs;
    ++heapAllocs;
    if (count > 1) {
      ++heap_array;
    }
    ++heap[typeId];
  }

  inline void incStackAlloc(int typeId, size_t count) {
    ++curStackAllocs;
    ++stackAllocs;
    if (count > 1) {
      ++stack_array;
    }
    ++stack[typeId];
  }

  inline void incGlobalAlloc(int typeId, size_t count) {
    ++globalAllocs;
    if (count > 1) {
      ++global_array;
    }
    ++global[typeId];
  }

  inline void decHeapAlloc() {
    if (curHeapAllocs > maxHeapAllocs) {
      maxHeapAllocs = curHeapAllocs;
    }
    --curHeapAllocs;
  }

  inline void decStackAlloc(size_t amount) {
    if (curStackAllocs > maxStackAllocs) {
      maxStackAllocs = curStackAllocs;
    }
    curStackAllocs -= amount;
  }

  inline void incUsedInRequest(const void* addr) {
    ++addrChecked;
    seen.insert(addr);
  }

  inline void incAddrReuse() {
    ++addrReuses;
  }

  inline void incAddrMissing(const void* addr) {
    ++addrMissing;
    missing.insert(addr);
  }

  void printStats() const {
    std::string s;
    llvm::raw_string_ostream buf(s);
    auto estMemConsumption = (maxHeapAllocs + maxStackAllocs) * memPerEntry;
    estMemConsumption += (maxStackAllocs * memInStack);
    estMemConsumption += (vectorSize + mapSize);
    estMemConsumption += mapNodeSizeInBytes * maxHeapAllocs;
    auto estMemConsumptionKByte = estMemConsumption / 1024.0;

    const auto getStr = [&](const auto memConsKB) {
      auto memStr = std::to_string(memConsKB);
      return memStr.substr(0, memStr.find('.') + 2);
    };

    buf << "------------\nAlloc Stats from softcounters\n"
        << "Total Calls .onAlloc [heap]:\t" << heapAllocs << " / " << heap_array << " arrays \n"
        << "Total Calls .onAlloc [stack]:\t" << stackAllocs << " / " << stack_array << " arrays \n"
        << "Total Calls .onAlloc [global]:\t" << globalAllocs << " / " << global_array << " arrays \n"
        << "Max. Heap Allocs:\t\t" << maxHeapAllocs << "\n"
        << "Max. Stack Allocs:\t\t" << maxStackAllocs << "\n"
        << "Addresses re-used:\t\t" << addrReuses << "\n"
        << "Addresses missed:\t\t" << addrMissing << "\n"
        << "Distinct Addresses checked:\t" << seen.size() << "\n"
        << "Addresses checked:\t\t" << addrChecked << "\n"
        << "Distinct Addresses missed:\t" << missing.size() << "\n"
        << "Estimated mem consumption:\t" << estMemConsumption << " bytes = " << getStr(estMemConsumptionKByte)
        << " kiB\n"
        << "vector overhead: " << vectorSize << " bytes\tmap overhead: " << mapSize
        << " bytes\tnode overhead: " << mapNodeSizeInBytes << "\n";

    std::set<int> type_id_set;
    const auto fill_set = [&type_id_set](const auto& map) {
      for (const auto& [key, val] : map) {
        type_id_set.insert(key);
      }
    };
    fill_set(heap);
    fill_set(global);
    fill_set(stack);

    const auto count = [](const auto& map, auto id) {
      auto it = map.find(id);
      if (it != map.end()) {
        return it->second;
      }
      return 0ll;
    };

    buf << "Allocation type detail (heap, stack, global):\n";
    for (auto type_id : type_id_set) {
      buf << typeart_get_type_name(type_id) << ": " << count(heap, type_id) << ", " << count(stack, type_id) << ", "
          << count(global, type_id) << "\n";
    }

    LOG_MSG(buf.str());
  }

  static AccessRecorder& get() {
    static AccessRecorder instance;
    return instance;
  }

 private:
  AccessRecorder()                       = default;
  AccessRecorder(AccessRecorder& other)  = default;
  AccessRecorder(AccessRecorder&& other) = default;

  // const int memPerEntry = sizeof(PointerInfo) + sizeof(void*);  // Type-map key + value
  const int memPerEntry = sizeof(TypeArtRT::MapEntry);
#if defined(USE_BTREE) || defined(USE_ABSL)
  const int mapNodeSizeInBytes = -1;  // TODO support
#else
  // based on https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/include/bits/stl_tree.h#L218
  const int mapNodeSizeInBytes = sizeof(
      std::remove_pointer<std::map<int, void*>::iterator::_Link_type>::type);  // GNU STL tree node: 3 pointers, 1 color
#endif
  const int memInStack     = sizeof(void*);                  // Stack allocs
  const int vectorSize     = sizeof(TypeArtRT::Stack);       // Stack overhead
  const int mapSize        = sizeof(TypeArtRT::PointerMap);  // Map overhead
  long long heapAllocs     = 0;
  long long stackAllocs    = 0;
  long long globalAllocs   = 0;
  long long maxHeapAllocs  = 0;
  long long maxStackAllocs = 0;
  long long curHeapAllocs  = 0;
  long long curStackAllocs = 0;
  long long addrReuses     = 0;
  long long addrMissing    = 0;
  long long addrChecked    = 0;

  long long stack_array{0};
  long long heap_array{0};
  long long global_array{0};
  std::unordered_set<const void*> missing;
  std::unordered_set<const void*> seen;
  type_count_map stack;
  type_count_map heap;
  type_count_map global;
};

/**
 * Used for no-operations in counter methods when not using softcounters.
 */
class NoneRecorder {
 public:
  inline void incHeapAlloc(int, size_t) {
  }
  inline void incStackAlloc(int, size_t) {
  }
  inline void incGlobalAlloc(int, size_t) {
  }
  inline void incUsedInRequest(const void*) {
  }
  inline void decHeapAlloc() {
  }
  inline void decStackAlloc(size_t) {
  }
  inline void incAddrReuse() {
  }
  inline void incAddrMissing(const void*) {
  }
  inline void printStats() const {
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

#endif /* RUNTIME_COUNTER_H_ */

//
// Created by ahueck on 12.10.20.
//

#ifndef TYPEART_ACCESSCOUNTER_H
#define TYPEART_ACCESSCOUNTER_H

#include "RuntimeData.h"
#include "RuntimeInterface.h"
#include "support/Logger.h"
#include "support/Table.h"

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

  template <typename Recorder>
  friend void serialise(const Recorder& r, llvm::raw_ostream& buf);
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

namespace memory {
struct MemOverhead {
  static constexpr auto pointerMapSize = sizeof(RuntimeT::PointerMap);  // Map overhead
  static constexpr auto perNodeSizeMap =
      sizeof(std::remove_pointer<std::map<MemAddr, PointerInfo>::iterator::_Link_type>::type) +
      sizeof(RuntimeT::MapEntry);                                         // not applicable to btree
  static constexpr auto stackVectorSize  = sizeof(RuntimeT::Stack);       // Stack overhead
  static constexpr auto perNodeSizeStack = sizeof(RuntimeT::StackEntry);  // Stack allocs
  double stack{0};
  double map{0};
};
inline MemOverhead estimate(Counter stack_max, Counter heap_max, Counter global_max, const double scale = 1024.0) {
  MemOverhead mem;
  mem.stack = double(MemOverhead::stackVectorSize +
                     MemOverhead::perNodeSizeStack * std::max<size_t>(RuntimeT::StackReserve, stack_max)) /
              scale;
  mem.map =
      double(MemOverhead::pointerMapSize + MemOverhead::perNodeSizeMap * (stack_max + heap_max + global_max)) / scale;
  return mem;
}
}  // namespace memory

template <typename Recorder>
void serialise(const Recorder& r, llvm::raw_ostream& buf) {
  if constexpr (std::is_same_v<Recorder, NoneRecorder>) {
    return;
  } else {
    const auto memory_use = memory::estimate(r.maxStackAllocs, r.maxHeapAllocs, r.globalAllocs);

    Table t("Alloc Stats from softcounters");
    t.wrap_length = true;
    t.put(Row::make("Total heap", r.heapAllocs, r.heapArray));
    t.put(Row::make("Total stack", r.stackAllocs, r.stackArray));
    t.put(Row::make("Total global", r.globalAllocs, r.globalArray));
    t.put(Row::make("Max. Heap Allocs", r.maxHeapAllocs));
    t.put(Row::make("Max. Stack Allocs", r.maxStackAllocs));
    t.put(Row::make("Addresses checked", r.addrChecked));
    t.put(Row::make("Distinct Addresses checked", r.seen.size()));
    t.put(Row::make("Addresses re-used", r.addrReuses));
    t.put(Row::make("Addresses missed", r.addrMissing));
    t.put(Row::make("Distinct Addresses missed", r.missing.size()));
    t.put(Row::make("Total free heap", r.heapAllocsFree, r.heapArrayFree));
    t.put(Row::make("Total free stack", r.stackAllocsFree, r.stackArrayFree));
    t.put(Row::make("Null/Zero/NullZero Addr", r.nullAlloc, r.zeroAlloc, r.nullAndZeroAlloc));
    t.put(Row::make("User-def. types", r.numUDefTypes));
    t.put(Row::make("Estimated memory use (KiB)", size_t(std::round(memory_use.map + memory_use.stack))));
    t.put(Row::make("Bytes per node map/stack", memory::MemOverhead::perNodeSizeMap,
                    memory::MemOverhead::perNodeSizeStack));

    t.print(buf);

    std::set<int> type_id_set;
    const auto fill_set = [&type_id_set](const auto& map) {
      for (const auto& [key, val] : map) {
        type_id_set.insert(key);
      }
    };
    fill_set(r.heapAlloc);
    fill_set(r.globalAlloc);
    fill_set(r.stackAlloc);
    fill_set(r.heapFree);
    fill_set(r.stackFree);

    const auto count = [](const auto& map, auto id) {
      auto it = map.find(id);
      if (it != map.end()) {
        return it->second;
      }
      return 0ll;
    };

    Table type_table("Allocation type detail (heap, stack, global)");
    type_table.table_header = '#';
    for (auto type_id : type_id_set) {
      type_table.put(Row::make(std::to_string(type_id), count(r.heapAlloc, type_id), count(r.stackAlloc, type_id),
                               count(r.globalAlloc, type_id), typeart_get_type_name(type_id)));
    }

    type_table.print(buf);

    Table type_table_free("Free allocation type detail (heap, stack)");
    type_table_free.table_header = '#';
    for (auto type_id : type_id_set) {
      type_table_free.put(Row::make(std::to_string(type_id), count(r.heapFree, type_id), count(r.stackFree, type_id),
                                    typeart_get_type_name(type_id)));
    }

    type_table_free.print(buf);
  }
}

}  // namespace softcounter

#if ENABLE_SOFTCOUNTER == 1
using Recorder = softcounter::AccessRecorder;
#else
using Recorder = softcounter::NoneRecorder;
#endif

}  // namespace typeart

#endif  // TYPEART_ACCESSCOUNTER_H

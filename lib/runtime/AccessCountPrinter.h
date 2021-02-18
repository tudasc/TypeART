//
// Created by ahueck on 30.12.20.
//

#ifndef TYPEART_ACCESSCOUNTPRINTER_H
#define TYPEART_ACCESSCOUNTPRINTER_H

#include "AccessCounter.h"
#include "support/Logger.h"
#include "support/Table.h"

#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace typeart::softcounter {
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
    const auto memory_use = memory::estimate(r.getMaxStackAllocs(), r.getMaxHeapAllocs(), r.getGlobalAllocs());

    Table t("Alloc Stats from softcounters");
    t.wrap_length = true;
    t.put(Row::make("Total heap", r.getHeapAllocs(), r.getHeapArray()));
    t.put(Row::make("Total stack", r.getStackAllocs(), r.getStackArray()));
    t.put(Row::make("Total global", r.getGlobalAllocs(), r.getGlobalArray()));
    t.put(Row::make("Max. Heap Allocs", r.getMaxHeapAllocs()));
    t.put(Row::make("Max. Stack Allocs", r.getMaxStackAllocs()));
    t.put(Row::make("Addresses checked", r.getAddrChecked()));
    t.put(Row::make("Distinct Addresses checked", r.getSeen().size()));
    t.put(Row::make("Addresses re-used", r.getAddrReuses()));
    t.put(Row::make("Addresses missed", r.getAddrMissing()));
    t.put(Row::make("Distinct Addresses missed", r.getMissing().size()));
    t.put(Row::make("Total free heap", r.getHeapAllocsFree(), r.getHeapArrayFree()));
    t.put(Row::make("Total free stack", r.getStackAllocsFree(), r.getStackArrayFree()));
    t.put(Row::make("OMP Stack/Heap/Free", r.getOmpStackCalls(), r.getOmpHeapCalls(), r.getOmpFreeCalls()));
    t.put(Row::make("Null/Zero/NullZero Addr", r.getNullAlloc(), r.getZeroAlloc(), r.getNullAndZeroAlloc()));
    t.put(Row::make("User-def. types", r.getNumUDefTypes()));
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
    fill_set(r.getHeapAlloc());
    fill_set(r.getGlobalAlloc());
    fill_set(r.getStackAlloc());
    fill_set(r.getHeapFree());
    fill_set(r.getStackFree());

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
      type_table.put(Row::make(std::to_string(type_id), count(r.getHeapAlloc(), type_id),
                               count(r.getStackAlloc(), type_id), count(r.getGlobalAlloc(), type_id),
                               typeart_get_type_name(type_id)));
    }

    type_table.print(buf);

    Table type_table_free("Free allocation type detail (heap, stack)");
    type_table_free.table_header = '#';
    for (auto type_id : type_id_set) {
      type_table_free.put(Row::make(std::to_string(type_id), count(r.getHeapFree(), type_id),
                                    count(r.getStackFree(), type_id), typeart_get_type_name(type_id)));
    }

    type_table_free.print(buf);

    auto numThreads = r.getNumThreads();

    std::stringstream ss;
    ss << "Per-thread counter values (" << numThreads << " threads)";
    Table thread_table(ss.str());

    auto putVals = [&thread_table](std::string name, const std::vector<Counter>& vals) {
      Row row(name);
      for (auto& val : vals) {
        row.put(Cell(val));
      }
      thread_table.put(std::move(row));
    };

    constexpr int printThreshold = 16;

    if (numThreads <= printThreshold) {
      putVals("Thread Heap Allocs", r.getHeapAllocsThreadData());
      putVals("Thread Heap Arrays", r.getHeapArrayThreadData());
      putVals("Thread Heap Allocs Free", r.getHeapAllocsFreeThreadData());
      putVals("Thread Heap Arrays Free", r.getHeapArrayFreeThreadData());
      putVals("Thread Stack Allocs", r.getStackAllocsThreadData());
      putVals("Thread Stack Arrays", r.getStackArrayThreadData());
      putVals("Thread Max. Stack Allocs", r.getMaxStackAllocsThreadData());
      putVals("Thread Stack Allocs Free", r.getStackAllocsFreeThreadData());
      putVals("Thread Stack Array Free", r.getStackArrayFreeThreadData());
    } else {
      thread_table.put(Row::make("Too many threads, output omitted."));
    }
    thread_table.print(buf);

#ifdef WRITE_THREAD_COUNTERS
    auto threadsFile = "typeart-thread-counters.csv";
    std::ofstream fout(threadsFile);

    auto putLine = [&fout](const std::vector<size_t>& vals) {
      if (vals.empty()) {
        fout << std::endl;
        return;
      }
      for (unsigned i = 0; i < vals.size() - 1; i++) {
        fout << vals[i] << ",";
      }
      fout << vals.back() << std::endl;
    };

    if (fout) {
      fout << "\'Heap Allocs\',"
           << "\'Heap Arrays\',"
           << "\'Heap Allocs Free\',"
           << "\'Heap Array Free\',"
           << "\'Stack Allocs\',"
           << "\'Stack Arrays\',"
           << "\'Max. Stack Allocs\',"
           << "\'Stack Allocs Free\',"
           << "\'Stack Array Free\'" << std::endl;

      auto threadIds = r.getThreadIds();
      for (auto id : threadIds) {
        std::vector<size_t> vals;
        auto threadData = r.getThreadDataAsMap(id);
        vals.push_back(threadData["heapAllocs"]);
        vals.push_back(threadData["heapArray"]);
        vals.push_back(threadData["heapAllocsFree"]);
        vals.push_back(threadData["heapArrayFree"]);
        vals.push_back(threadData["stackAllocs"]);
        vals.push_back(threadData["stackArray"]);
        vals.push_back(threadData["maxStackAllocs"]);
        vals.push_back(threadData["stackAllocsFree"]);
        vals.push_back(threadData["stackArrayFree"]);
        putLine(vals);
      }

    } else {
      LOG_ERROR("Unable to write thread counters to file " << threadsFile)
    }
#endif
  }
}
}  // namespace typeart::softcounter

#endif  // TYPEART_ACCESSCOUNTPRINTER_H

// TypeART library
//
// Copyright (c) 2017-2025 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
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
#include <utility>

namespace typeart::softcounter {
namespace memory {
struct MemOverhead {
  static constexpr auto pointerMapSize = sizeof(RuntimeT::PointerMap);  // Map overhead
  static constexpr auto perNodeSizeMap =
      64U + sizeof(RuntimeT::MapEntry);  // rough estimate, not applicable to btree; 64U is internal node size
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
void serialize(const Recorder& r, std::ostringstream& buf) {
  if constexpr (std::is_same_v<Recorder, NoneRecorder>) {
    return;
  } else {
    // const auto memory_use = memory::estimate(r.getMaxStackAllocs(), r.getMaxHeapAllocs(), r.getGlobalAllocs());
    Table overview_table("Alloc Stats from softcounters");
    overview_table.wrap_length_ = true;
    overview_table.put(Row::make("Total heap", r.getHeapAllocs(), r.getHeapArray()));
    overview_table.put(Row::make("Total stack", r.getStackAllocs(), r.getStackArray()));
    overview_table.put(Row::make("Total global", r.getGlobalAllocs(), r.getGlobalArray()));
    overview_table.put(Row::make("Max. heap", r.getMaxHeapAllocs()));
    overview_table.put(Row::make("Max. stack", r.getMaxStackAllocs()));
    overview_table.put(Row::make("Addresses checked", r.getAddrChecked()));
    overview_table.put(Row::make("Distinct addresses checked", r.getSeen().size()));
    overview_table.put(Row::make("Addresses re-used", r.getAddrReuses()));
    overview_table.put(Row::make("Addresses missed", r.getAddrMissing()));
    overview_table.put(Row::make("Distinct addresses missed", r.getMissing().size()));
    overview_table.put(Row::make("Total free heap", r.getHeapAllocsFree(), r.getHeapArrayFree()));
    overview_table.put(Row::make("Total free stack", r.getStackAllocsFree(), r.getStackArrayFree()));
    overview_table.put(
        Row::make("OMP stack/heap/free", r.getOmpStackCalls(), r.getOmpHeapCalls(), r.getOmpFreeCalls()));
    overview_table.put(
        Row::make("Null/Zero/NullZero addr", r.getNullAlloc(), r.getZeroAlloc(), r.getNullAndZeroAlloc()));
    overview_table.put(Row::make("User-def. types", r.getNumUDefTypes()));
    overview_table.put(Row::make("Distinct query types", r.getTypeQuery().size()));
    // t.put(Row::make("Estimated memory use (KiB)", size_t(std::round(memory_use.map + memory_use.stack))));
    // t.put(Row::make("Bytes per node map/stack", memory::MemOverhead::perNodeSizeMap,
    // memory::MemOverhead::perNodeSizeStack));

    overview_table.print(buf);

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
      return 0LL;
    };
    {
      Table type_table("Allocation type detail (heap, stack, global)");
      type_table.table_header_ = '#';
      for (auto type_id : type_id_set) {
        type_table.put(Row::make(std::to_string(type_id), count(r.getHeapAlloc(), type_id),
                                 count(r.getStackAlloc(), type_id), count(r.getGlobalAlloc(), type_id),
                                 typeart_get_type_name(type_id)));
      }
      type_table.print(buf);
    }
    {
      Table type_table_free("Free allocation type detail (heap, stack)");
      type_table_free.table_header_ = '#';
      for (auto type_id : type_id_set) {
        const auto count_type_h = count(r.getHeapFree(), type_id);
        const auto count_type_s = count(r.getStackFree(), type_id);
        if (count_type_h == 0 && count_type_s == 0) {
          continue;
        }
        type_table_free.put(
            Row::make(std::to_string(type_id), count_type_h, count_type_s, typeart_get_type_name(type_id)));
      }

      type_table_free.print(buf);
    }
    {
      Table type_table_query("Query type detail");
      type_table_query.table_header_ = '#';
      const auto& query_map          = r.getTypeQuery();
      std::set<int> query_id_set;
      for (const auto& [key, val] : query_map) {
        query_id_set.insert(key);
      }
      for (auto query_type_id : query_id_set) {
        const auto count_type = count(query_map, query_type_id);
        type_table_query.put(
            Row::make(std::to_string(query_type_id), count_type, typeart_get_type_name(query_type_id)));
      }

      type_table_query.print(buf);
    }

    const auto numThreads = r.getNumThreads();
    if (numThreads < 2) {
      return;
    }
    std::stringstream ss;
    ss << "Per-thread counter values (" << numThreads << " threads)";
    Table thread_table(ss.str());
    thread_table.table_header_ = '#';

    auto print_thread_row = [&thread_table](std::string name, const std::vector<Counter>& vals) {
      Row row(std::move(name));
      for (const auto& val : vals) {
        row.put(Cell(val));
      }
      thread_table.put(std::move(row));
    };

    print_thread_row("Thread Heap Allocs", r.getHeapAllocsThreadData());
    print_thread_row("Thread Heap Arrays", r.getHeapArrayThreadData());
    print_thread_row("Thread Heap Allocs Free", r.getHeapAllocsFreeThreadData());
    print_thread_row("Thread Heap Arrays Free", r.getHeapArrayFreeThreadData());
    print_thread_row("Thread Stack Allocs", r.getStackAllocsThreadData());
    print_thread_row("Thread Stack Arrays", r.getStackArrayThreadData());
    print_thread_row("Thread Max. Stack Allocs", r.getMaxStackAllocsThreadData());
    print_thread_row("Thread Stack Allocs Free", r.getStackAllocsFreeThreadData());
    print_thread_row("Thread Stack Array Free", r.getStackArrayFreeThreadData());
    thread_table.print(buf);
  }
}
}  // namespace typeart::softcounter

#endif  // TYPEART_ACCESSCOUNTPRINTER_H

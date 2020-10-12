//
// Created by ahueck on 12.10.20.
//

#ifndef TYPEART_ACCESSCOUNTER_H
#define TYPEART_ACCESSCOUNTER_H

#include "Logger.h"
#include "RuntimeData.h"
#include "RuntimeInterface.h"

#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace typeart {
namespace softcounter {

class AccessRecorder {
 public:
  using TypeCountMap = std::unordered_map<int, long long>;
  ~AccessRecorder()  = default;

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

  static AccessRecorder& get() {
    static AccessRecorder instance;
    return instance;
  }

 private:
  AccessRecorder()                       = default;
  AccessRecorder(AccessRecorder& other)  = default;
  AccessRecorder(AccessRecorder&& other) = default;

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
  TypeCountMap stack;
  TypeCountMap heap;
  TypeCountMap global;

  template <typename Recorder>
  friend void serialise(const Recorder& r, llvm::raw_ostream& buf);
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

struct Cell {
  enum align_as { left, right };

  explicit Cell(std::string v) : c(v), w(c.size()), align(left) {
    w     = c.size();
    align = left;
  }

  explicit Cell(const char* v) : Cell(std::string(v)) {
  }

  template <typename number>
  explicit Cell(number v) : Cell(std::to_string(v)) {
    align = right;
  }

  std::string c;
  unsigned w;
  align_as align;
};

struct Row {
  Cell id;
  std::vector<Cell> cells;

  explicit Row(std::string name) : id(name) {
  }

  Row& put(Cell c) {
    cells.emplace_back(c);
    return *this;
  }

  static Row make_row(std::string name) {
    auto r = Row(name);
    return r;
  }

  template <typename... Cells>
  static Row make(std::string name, Cells... c) {
    auto r = Row(name);
    (r.put(Cell(c)), ...);
    return r;
  }
};

struct Table {
  std::string name{"Table"};
  std::vector<Row> row_vec;
  unsigned columns{1};
  unsigned rows{0};
  unsigned max_row_label_width{1};
  char separator{'-'};

  explicit Table(std::string name) : name(name) {
  }

  void put(Row r) {
    row_vec.emplace_back(r);
    columns = std::max<unsigned>(columns, r.cells.size());
    ++rows;
    max_row_label_width = std::max<unsigned>(max_row_label_width, r.id.w);
  }

  void print(llvm::raw_ostream& s) {
    auto max_row_id = max_row_label_width + 1;

    // determine per column width
    std::vector<unsigned> col_width(columns, 4);
    for (const auto& row : row_vec) {
      unsigned col_num{0};
      for (const auto& col : row.cells) {
        col_width[col_num] = (std::max<unsigned>(col_width[col_num], col.w + 1));
        ++col_num;
      }
    }

    s << std::string(std::max<unsigned>(max_row_id, name.size()), separator) << "\n";
    s << name;
    s << "\n";
    for (const auto& row : row_vec) {
      s << llvm::left_justify(row.id.c, max_row_id) << ":";

      if (row.cells.empty()) {
        s << "\n";
        continue;
      }
      unsigned col_num{0};
      auto num_beg = std::begin(row.cells);
      s << llvm::right_justify(num_beg->c, col_width[col_num]);
      std::for_each(std::next(num_beg), std::end(row.cells), [&](const auto& v) {
        const auto width   = col_width[++col_num];
        const auto aligned = v.align == Cell::right ? llvm::right_justify(v.c, width) : llvm::left_justify(v.c, width);
        s << " , " << aligned;
      });
      s << "\n";
    }
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
inline MemOverhead estimate(long long stack_max, long long heap_max, long long global_max,
                            const double scale = 1024.0) {
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
  }

  const auto memory_use = memory::estimate(r.maxStackAllocs, r.maxHeapAllocs, r.globalAllocs);

  Table t("Alloc Stats from softcounters");
  t.put(Row::make("Total heap", r.heapAllocs, r.heap_array));
  t.put(Row::make("Total stack", r.stackAllocs, r.stack_array));
  t.put(Row::make("Total global", r.globalAllocs, r.global_array));
  t.put(Row::make("Max. Heap Allocs", r.maxHeapAllocs));
  t.put(Row::make("Max. Stack Allocs", r.maxStackAllocs));
  t.put(Row::make("Addresses checked", r.addrChecked));
  t.put(Row::make("Distinct Addresses checked", r.seen.size()));
  t.put(Row::make("Addresses re-used", r.addrReuses));
  t.put(Row::make("Addresses missed", r.addrMissing));
  t.put(Row::make("Distinct Addresses missed", r.missing.size()));
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
  fill_set(r.heap);
  fill_set(r.global);
  fill_set(r.stack);

  const auto count = [](const auto& map, auto id) {
    auto it = map.find(id);
    if (it != map.end()) {
      return it->second;
    }
    return 0ll;
  };

  Table type_table("Allocation type detail (heap, stack, global)");
  type_table.separator = '#';
  for (auto type_id : type_id_set) {
    type_table.put(Row::make(std::to_string(type_id), count(r.heap, type_id), count(r.stack, type_id),
                             count(r.global, type_id), typeart_get_type_name(type_id)));
  }

  type_table.print(buf);

  buf.flush();
}

}  // namespace softcounter

#if ENABLE_SOFTCOUNTER == 1
using Recorder = softcounter::AccessRecorder;
#else
using Recorder = softcounter::NoneRecorder;
#endif

}  // namespace typeart

#endif  // TYPEART_ACCESSCOUNTER_H

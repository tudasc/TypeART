//
// Created by ahueck on 12.10.20.
//

#ifndef TYPEART_ACCESSCOUNTER_H
#define TYPEART_ACCESSCOUNTER_H

#include "Logger.h"
#include "Runtime.h"
#include "RuntimeInterface.h"

#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace typeart {
namespace softcounter {

class AccessRecorder {
 public:
  using type_count_map = std::unordered_map<int, long long>;
  ~AccessRecorder()    = default;

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
  type_count_map stack;
  type_count_map heap;
  type_count_map global;

  template <typename Recorder>
  friend std::string serialise(const Recorder& r);
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
  template <typename number>
  Cell(number v) : c(std::to_string(v)) {
    w = c.size();
  }
  Cell(std::string v) : c(v) {
    w = c.size();
  }
  std::string c;
  unsigned w;
};

struct Row {
  Row(std::string name) : id(name) {
  }
  Cell id;
  std::vector<Cell> numbers;

  Row& put(Cell c) {
    numbers.emplace_back(c);
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

  Table(std::string name) : name(name) {
  }

  void put(Row r) {
    row_vec.emplace_back(r);
    columns = std::max<unsigned>(columns, r.numbers.size());
    ++rows;
  }

  void print(llvm::raw_ostream& s) {
    auto max_it     = std::max_element(std::begin(row_vec), std::end(row_vec),
                                   [](const auto& a, const auto& b) { return a.id.w < b.id.w; });
    auto max_row_id = max_it->id.w + 1;

    s << std::string(std::max<unsigned>(max_row_id, name.size()), '-') << "\n";
    s << name;
    s << "\n";
    for (const auto& row : row_vec) {
      s << llvm::left_justify(row.id.c, max_row_id) << ":";

      if (row.numbers.empty()) {
        s << "\n";
        continue;
      }
      auto num_beg = std::begin(row.numbers);
      s << llvm::right_justify(num_beg->c, 8);
      std::for_each(std::next(num_beg), std::end(row.numbers), [&s](const auto& v) {
        s << " , " << llvm::right_justify(v.c, 8);
        ;
      });
      s << "\n";
    }
  }
};

template <typename Recorder>
std::string serialise(const Recorder& r) {
  if constexpr (std::is_same_v<Recorder, NoneRecorder>) {
    return "";
  }

  std::string s;
  llvm::raw_string_ostream buf(s);
  //  auto estMemConsumption = (r.maxHeapAllocs + r.maxStackAllocs) * r.memPerEntry;
  //  estMemConsumption += (r.maxStackAllocs * r.memInStack);
  //  estMemConsumption += (r.vectorSize + r.mapSize);
  //  estMemConsumption += r.mapNodeSizeInBytes * r.maxHeapAllocs;
  //  auto estMemConsumptionKByte = estMemConsumption / 1024.0;

  const auto getStr = [&](const auto memConsKB) {
    auto memStr = std::to_string(memConsKB);
    return memStr.substr(0, memStr.find('.') + 2);
  };

  Table t("Alloc Stats from softcounters");
  t.put(Row::make_row("Total heap").put({r.heapAllocs}).put({r.heap_array}));
  t.put(Row::make_row("Total stack").put({r.stackAllocs}).put({r.stack_array}));
  t.put(Row::make("Total global", r.globalAllocs, r.global_array));
  t.put(Row::make("Max. Heap Allocs", r.maxHeapAllocs));
  t.put(Row::make("Max. Stack Allocs", r.maxStackAllocs));
  t.put(Row::make("Addresses re-used", r.addrReuses));
  t.put(Row::make("Addresses missed", r.addrMissing));
  t.put(Row::make("Distinct Addresses checked", r.seen.size()));
  t.put(Row::make("Addresses checked", r.addrChecked));
  t.put(Row::make("Distinct Addresses missed", r.missing.size()));
  //      << "Estimated mem consumption:\t" << estMemConsumption << " bytes = " << getStr(estMemConsumptionKByte)
  //      << " kiB\n"
  //      << "vector overhead: " << r.vectorSize << " bytes\tmap overhead: " << r.mapSize
  //      << " bytes\tnode overhead: " << r.mapNodeSizeInBytes << "\n";

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

  buf << "Allocation type detail (heap, stack, global):\n";
  for (auto type_id : type_id_set) {
    buf << typeart_get_type_name(type_id) << ": " << count(r.heap, type_id) << ", " << count(r.stack, type_id) << ", "
        << count(r.global, type_id) << "\n";
  }

  return buf.str();
}

}  // namespace softcounter

#if ENABLE_SOFTCOUNTER == 1
using Recorder = softcounter::AccessRecorder;
#else
using Recorder = softcounter::NoneRecorder;
#endif

}  // namespace typeart

#endif  // TYPEART_ACCESSCOUNTER_H

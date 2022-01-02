// TypeART library
//
// Copyright (c) 2017-2022 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#ifndef TYPEART_TABLE_H
#define TYPEART_TABLE_H

#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace typeart {

namespace detail {
template <typename Number>
inline std::string num2str(Number num) {
  if constexpr (std::is_floating_point_v<Number>) {
    std::ostringstream os;
    os << std::setprecision(2) << std::fixed << num;

    return os.str();
  } else {
    return std::to_string(num);
  }
}

struct JustifiedString final {
  enum align_as { left, right };
  std::string_view string;
  int width;
  align_as alignment;

 public:
  JustifiedString(std::string_view string, int w, align_as a) : string(string), width(w), alignment(a) {
  }

  friend std::ostream& operator<<(std::ostream& os, const JustifiedString& js) {
    os << std::setw(js.width) << (js.alignment == JustifiedString::align_as::left ? std::left : std::right)
       << js.string;
    return os;
  }
};
}  // namespace detail

struct Cell {
  enum align_as { left, right };

  std::string c;
  int w;
  align_as align;

  explicit Cell(std::string v) : c(std::move(v)), w(c.size()), align(left) {
    w     = c.size();
    align = left;
  }

  explicit Cell(const char* v) : Cell(std::string(v)) {
  }

  template <typename number>
  explicit Cell(number v) : Cell(detail::num2str(v)) {
    align = right;
  }
};

struct Row {
  Cell label;
  std::vector<Cell> cells;

  explicit Row(std::string name) : label(std::move(name)) {
  }

  Row& put(Cell&& c) {
    cells.emplace_back(c);
    return *this;
  }

  static Row make_row(std::string name) {
    auto r = Row(std::move(name));
    return r;
  }

  template <typename... Cells>
  static Row make(std::string name, Cells&&... c) {
    auto r = make_row(std::move(name));
    (r.put(Cell(c)), ...);
    return r;
  }
};

struct Table {
  std::string title{"Table"};
  std::vector<Row> row_vec;
  int columns{0};
  int rows{0};
  int max_row_label_width{1};
  std::string cell_sep{" , "};
  std::string empty_cell{"-"};
  char table_header{'-'};
  bool wrap_header{false};
  bool wrap_length{false};
  bool colon_empty{false};

  explicit Table(std::string title) : title(std::move(title)) {
  }

  void put(Row&& r) {
    row_vec.emplace_back(r);
    columns = std::max<int>(columns, r.cells.size());
    ++rows;
    max_row_label_width = std::max<int>(max_row_label_width, r.label.w);
  }

  template <typename... Rows>
  static Table make(std::string title, Rows&&... r) {
    Table t{std::move(title)};
    (t.put(std::forward<Rows>(r)), ...);
    return t;
  }

  void print(std::ostream& s) const {
    const auto left_justify = [](std::string_view cell_str, int width) {
      using namespace detail;
      return JustifiedString(cell_str, width, JustifiedString::left);
    };

    const auto right_justify = [](std::string_view cell_str, int width) {
      using namespace detail;
      return JustifiedString(cell_str, width, JustifiedString::right);
    };

    const auto max_row_id = max_row_label_width + 1;

    // determine per column width
    std::vector<int> col_width(columns, 4);
    for (const auto& row : row_vec) {
      int col_num{0};
      for (const auto& col : row.cells) {
        col_width[col_num] = std::max<int>(col_width[col_num], col.w + 1);
        ++col_num;
      }
    }

    // how long is the header separation supposed to be
    const auto head_rep_count = [&]() -> int {
      if (wrap_length) {
        const int sum_col_width = std::accumulate(std::begin(col_width), std::end(col_width), 0);
        const int title_width   = std::max<int>(max_row_id, title.size());

        return title_width + sum_col_width + columns * cell_sep.size();
      }
      return std::max<int>(max_row_id, title.size());
    };

    s << std::string(head_rep_count(), table_header) << "\n";
    s << title << "\n";
    for (const auto& row : row_vec) {
      s << left_justify(row.label.c, max_row_id);

      if (!row.cells.empty() || colon_empty) {
        s << ":";
      }

      int col_num{0};
      auto num_beg         = std::begin(row.cells);
      const auto has_cells = num_beg != std::end(row.cells);

      if (has_cells) {
        // print first row cell, then subsequent cells with *cell_sep* as prefix
        s << right_justify(num_beg->c, col_width[col_num]);
        std::for_each(std::next(num_beg), std::end(row.cells), [&](const auto& v) {
          const auto width        = col_width[++col_num];
          const auto aligned_cell = v.align == Cell::right ? right_justify(v.c, width) : left_justify(v.c, width);
          s << cell_sep << aligned_cell;
        });
      }

      // fill up empty columns with empty_cell
      const int padding = columns - col_num - 1;
      if (padding > 0 && !empty_cell.empty()) {
        const auto iterate_w = [&]() -> int {
          const auto width = col_width[++col_num];
          return width;
        };

        // print first empty padding, then subsequent padding with *cell_sep* as prefix
        if (!has_cells) {
          s << right_justify(empty_cell, iterate_w());
        }
        for (int pad = 0; pad < padding; ++pad) {
          s << cell_sep << right_justify(empty_cell, iterate_w());
        }
      }
      s << "\n";
    }

    if (wrap_header) {
      // Put header separation at bottom too
      s << std::string(head_rep_count(), table_header) << "\n";
    }

    s.flush();
  }
};
}  // namespace typeart
#endif  // TYPEART_TABLE_H

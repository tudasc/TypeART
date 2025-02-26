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

#ifndef TYPEART_TABLE_H
#define TYPEART_TABLE_H

#include <algorithm>
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
  std::string_view string_;
  int width_;
  align_as alignment_;

 public:
  JustifiedString(std::string_view string, int width, align_as alignment)
      : string_(string), width_(width), alignment_(alignment) {
  }

  friend std::ostream& operator<<(std::ostream& os, const JustifiedString& js) {
    os << std::setw(js.width_) << (js.alignment_ == JustifiedString::align_as::left ? std::left : std::right)
       << js.string_;
    return os;
  }
};
}  // namespace detail

struct Cell {
  enum align_as { kLeft, kRight };

  std::string cell_value_;
  int width_;
  align_as alignment_;

  explicit Cell(std::string value) : cell_value_(std::move(value)), width_(cell_value_.size()), alignment_(kLeft) {
    width_     = cell_value_.size();
    alignment_ = kLeft;
  }

  explicit Cell(const char* v) : Cell(std::string(v)) {
  }

  template <typename number>
  explicit Cell(number v) : Cell(detail::num2str(v)) {
    alignment_ = kRight;
  }
};

struct Row {
  Cell label_;
  std::vector<Cell> cells_;

  explicit Row(std::string name) : label_(std::move(name)) {
  }

  Row& put(Cell&& cell) {
    cells_.emplace_back(cell);
    return *this;
  }

  static Row make_row(std::string name) {
    auto row = Row(std::move(name));
    return row;
  }

  template <typename... Cells>
  static Row make(std::string name, Cells&&... cells) {
    auto row = make_row(std::move(name));
    (row.put(Cell(cells)), ...);
    return row;
  }
};

struct Table {
  std::string title_{"Table"};
  std::vector<Row> row_vec_;
  int columns_{0};
  int rows_{0};
  int max_row_label_width_{1};
  std::string cell_sep_{" , "};
  std::string empty_cell_{"-"};
  char table_header_{'-'};
  bool wrap_header_{false};
  bool wrap_length_{false};
  bool colon_empty_{false};

  explicit Table(std::string title) : title_(std::move(title)) {
  }

  void put(Row&& row) {
    row_vec_.emplace_back(row);
    columns_ = std::max<int>(columns_, row.cells_.size());
    ++rows_;
    max_row_label_width_ = std::max<int>(max_row_label_width_, row.label_.width_);
  }

  template <typename... Rows>
  static Table make(std::string title, Rows&&... rows) {
    Table t{std::move(title)};
    (t.put(std::forward<Rows>(rows)), ...);
    return t;
  }

  void print(std::ostream& out_stream) const {
    const auto left_justify = [](std::string_view cell_str, int width) {
      using namespace detail;
      return JustifiedString(cell_str, width, JustifiedString::left);
    };

    const auto right_justify = [](std::string_view cell_str, int width) {
      using namespace detail;
      return JustifiedString(cell_str, width, JustifiedString::right);
    };

    const auto max_row_id = max_row_label_width_ + 1;

    // determine per column width
    std::vector<int> col_width(columns_, 4);
    for (const auto& row : row_vec_) {
      int col_num{0};
      for (const auto& col : row.cells_) {
        col_width[col_num] = std::max<int>(col_width[col_num], col.width_ + 1);
        ++col_num;
      }
    }

    // how long is the header separation supposed to be
    const auto head_rep_count = [&]() -> int {
      if (wrap_length_) {
        const int sum_col_width = std::accumulate(std::begin(col_width), std::end(col_width), 0);
        const int title_width   = std::max<int>(max_row_id, title_.size());

        return title_width + sum_col_width + columns_ * cell_sep_.size();
      }
      return std::max<int>(max_row_id, title_.size());
    };

    out_stream << std::string(head_rep_count(), table_header_) << "\n";
    out_stream << title_ << "\n";
    for (const auto& row : row_vec_) {
      out_stream << left_justify(row.label_.cell_value_, max_row_id);

      if (!row.cells_.empty() || colon_empty_) {
        out_stream << ":";
      }

      int col_num{0};
      auto num_beg         = std::begin(row.cells_);
      const auto has_cells = num_beg != std::end(row.cells_);

      if (has_cells) {
        // print first row cell, then subsequent cells with *cell_sep* as prefix
        out_stream << right_justify(num_beg->cell_value_, col_width[col_num]);
        std::for_each(std::next(num_beg), std::end(row.cells_), [&](const Cell& v) {
          const auto width = col_width[++col_num];
          const auto aligned_cell =
              v.alignment_ == Cell::kRight ? right_justify(v.cell_value_, width) : left_justify(v.cell_value_, width);
          out_stream << cell_sep_ << aligned_cell;
        });
      }

      // fill up empty columns with empty_cell
      const int padding = columns_ - col_num - 1;
      if (padding > 0 && !empty_cell_.empty()) {
        const auto iterate_w = [&]() -> int {
          const auto width = col_width[++col_num];
          return width;
        };

        // print first empty padding, then subsequent padding with *cell_sep* as prefix
        if (!has_cells) {
          out_stream << right_justify(empty_cell_, iterate_w());
        }
        for (int pad = 0; pad < padding; ++pad) {
          out_stream << cell_sep_ << right_justify(empty_cell_, iterate_w());
        }
      }
      out_stream << "\n";
    }

    if (wrap_header_) {
      // Put header separation at bottom too
      out_stream << std::string(head_rep_count(), table_header_) << "\n";
    }

    out_stream.flush();
  }
};
}  // namespace typeart
#endif  // TYPEART_TABLE_H

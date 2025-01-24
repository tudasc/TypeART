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

#ifndef TYPEART_CONFIGURATION_H
#define TYPEART_CONFIGURATION_H

#include "support/Logger.h"

#include "llvm/ADT/StringMap.h"

#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace typeart::config {

namespace detail {
template <typename T, typename Types>
struct VariantContains;

template <typename T, typename... Ts>
struct VariantContains<T, std::variant<Ts...>> : std::bool_constant<(... || std::is_same<T, Ts>{})> {};

template <typename T, typename... Ts>
inline constexpr bool VariantContains_v = VariantContains<T, Ts...>::value;

template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;
}  // namespace detail

class OptionValue final {
  using Value = std::variant<std::string, int, double, bool>;
  Value value_;

 public:
  OptionValue() = default;

  OptionValue(std::string val) : value_(std::move(val)) {
  }

  OptionValue(const char* val) : OptionValue(std::string{val}) {
  }

  template <typename T>
  explicit OptionValue(T val) : value_(val) {
    static_assert(detail::VariantContains_v<T, Value>, "T is not compatible with OptionValue class.");
  }

  //  template <typename T>
  explicit operator std::string() const {
    std::string val{};
    std::visit(typeart::config::detail::overloaded{[&](const auto& v_val) { val = std::to_string(v_val); },  //
                                                   [&](const std::string& v_val) { val = v_val; }},
               value_);
    return val;
  }

  template <typename T>
  operator T() const {
    if constexpr (std::is_enum_v<T>) {
      const int* value = std::get_if<int>(&value_);
      if (nullptr == value) {
        return T{0};
      }
      return static_cast<T>(*value);
    } else {
      static_assert(detail::VariantContains_v<T, Value>, "T is not compatible with OptionValue class.");
      const auto* value = std::get_if<T>(&value_);
      if (nullptr == value) {
        return T{};
      }
      return *value;
    }
  }
};

using OptionsMap       = llvm::StringMap<OptionValue>;
using OptOccurrenceMap = llvm::StringMap<bool>;

class Configuration {
 public:
  [[nodiscard]] virtual std::optional<OptionValue> getValue(std::string_view opt_path) const = 0;

  [[nodiscard]] virtual OptionValue getValueOr(std::string_view opt_path, OptionValue alt) const {
    return getValue(opt_path).value_or(alt);
  }

  [[nodiscard]] virtual OptionValue operator[](std::string_view opt_path) const {
    return getValueOr(opt_path, config::OptionValue{});
  }

  virtual ~Configuration() = default;
};

}  // namespace typeart::config

#endif  // TYPEART_CONFIGURATION_H

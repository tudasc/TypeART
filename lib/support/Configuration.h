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

#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace typeart::config {

struct ConfigStdArgs final {
#define TYPEART_CONFIG_OPTION(name, path, type, def_value, description) static constexpr char name[] = path;
#include "ConfigurationBaseOptions.h"
#undef TYPEART_CONFIG_OPTION
};

struct ConfigStdArgValues final {
#define TYPEART_CONFIG_OPTION(name, path, type, def_value, description) \
  static constexpr decltype(def_value) name{def_value};
#include "ConfigurationBaseOptions.h"
#undef TYPEART_CONFIG_OPTION
};

struct ConfigStdArgTypes final {
#define TYPEART_CONFIG_OPTION(name, path, type, default_value, description) using name##_ty = type;
#include "ConfigurationBaseOptions.h"
#undef TYPEART_CONFIG_OPTION
};

struct ConfigStdArgDescriptions final {
#define TYPEART_CONFIG_OPTION(name, path, type, default_value, description) static constexpr char name[] = description;
#include "ConfigurationBaseOptions.h"
#undef TYPEART_CONFIG_OPTION
};

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

class Configuration {
 public:
  [[nodiscard]] virtual std::optional<OptionValue> getValue(std::string_view opt_path) const = 0;

  [[nodiscard]] virtual OptionValue getValueOr(std::string_view opt_path, OptionValue alt) const {
    const auto val = getValue(opt_path);
    if (val) {
      return val.value();
    }
    return alt;
  }

  [[nodiscard]] virtual OptionValue operator[](std::string_view opt_path) const {
    return getValueOr(opt_path, config::OptionValue{});
  }

  virtual ~Configuration() = default;
};

}  // namespace typeart::config

#endif  // TYPEART_CONFIGURATION_H

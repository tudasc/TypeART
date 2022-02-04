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

#ifndef LIB_UTIL_H_
#define LIB_UTIL_H_

//#include "Logger.h"

#include "compat/CallSite.h"

#include "llvm/Demangle/Demangle.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"

namespace typeart::util {

namespace detail {
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4502.pdf :
template <class>
struct type_sink {
  using type = void;
};
template <class T>
using type_sink_t = typename type_sink<T>::type;

#define has_member(_NAME_)                  \
  template <class T, class = void>          \
  struct has_##_NAME_ : std::false_type {}; \
  template <class T>                        \
  struct has_##_NAME_<T, type_sink_t<decltype(std::declval<T>()._NAME_())>> : std::true_type {};

// clang-format off
has_member(begin)
has_member(end)
#undef has_member

template <typename T>
using has_begin_end_t = typename std::integral_constant<bool,
      has_begin<T>{} && has_end<T>{}>::type;
// clang-format on

template <typename Val>
inline std::string dump(const Val& s, std::false_type) {
  std::string tmp;
  llvm::raw_string_ostream out(tmp);
  s.print(out);

  return tmp;
}

template <typename Val>
inline std::string dump(const Val& s, std::true_type) {
  auto beg = s.begin();
  auto end = s.end();
  if (beg == end) {
    return "[ ]";
  }

  std::string tmp;
  llvm::raw_string_ostream out(tmp);
  auto next = std::next(beg);
  out << "[ " << *(*(beg));
  std::for_each(next, end, [&out](auto v) { out << " , " << *v; });
  out << " ]";
  return out.str();
}

}  // namespace detail

template <typename Val>
inline std::string dump(const Val& s) {
  using namespace detail;
  return dump(s, has_begin_end_t<Val>{});
}

template <typename String>
inline std::string demangle(String&& s) {
  std::string name = std::string{s};
  auto demangle    = llvm::itaniumDemangle(name.data(), nullptr, nullptr, nullptr);
  if (demangle && !std::string(demangle).empty()) {
    return {demangle};
  }
  return name;
}

template <typename T>
inline std::string try_demangle(const T& site) {
  if constexpr (std::is_same_v<T, llvm::CallSite>) {
    if (site.isIndirectCall()) {
      return "";
    }
    return demangle(site.getCalledFunction()->getName());
  } else {
    if constexpr (std::is_same_v<T, llvm::Function>) {
      return demangle(site.getName());
    } else {
      return demangle(site);
    }
  }
}

template <typename Predicate>
inline std::vector<llvm::Instruction*> find_all(llvm::Function* f, Predicate&& p) {
  std::vector<llvm::Instruction*> v;
  for (auto& bb : *f) {
    for (auto& inst : bb) {
      if (p(inst)) {
        v.push_back(&inst);
      }
    }
  }
  return v;
}

template <typename Predicate>
inline llvm::Instruction* find_first_of(llvm::Function* f, Predicate&& p) {
  for (auto& bb : *f) {
    for (auto& inst : bb) {
      if (p(inst)) {
        return &inst;
      }
    }
  }
  return nullptr;
}

inline bool regex_matches(const std::string& regex, const std::string& in, bool case_sensitive = false) {
  using namespace llvm;
  Regex r(regex, !case_sensitive ? Regex::IgnoreCase : Regex::NoFlags);
  return r.match(in);
}

inline std::string glob2regex(const std::string& glob) {
  // Handles glob with no error checking:
  // Choice: '{a,b,c}' eq. (a|b|c)
  // Any char in bracket: '[abc]' eq. [abc]
  // Any char: '?' eq. .
  // Any string: '*' eq. .*
  std::string glob_reg{"^"};
  int in_curly{0};
  for (char c : glob) {
    switch (c) {
      case '?':
        glob_reg += ".";
        break;
      case '*':
        glob_reg += ".*";
        break;
      case '{':
        ++in_curly;
        glob_reg += "(";
        break;
      case '}':
        if (in_curly > 0) {
          --in_curly;
          glob_reg += ")";
        } else {
          glob_reg += c;
        }
        break;
      case ',':
        glob_reg += (in_curly > 0 ? "|" : ",");
        break;
      default:
        if (strchr("()^$|*+.\\", c) != nullptr) {
          glob_reg += '\\';
        }
        glob_reg += c;
        break;
    }
  }
  glob_reg += "$";
  return glob_reg;
}

}  // namespace typeart::util

#endif /* LIB_UTIL_H_ */

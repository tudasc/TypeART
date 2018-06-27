/*
 * Util.h
 *
 *  Created on: May 7, 2018
 *      Author: ahueck
 */

#ifndef LIB_UTIL_H_
#define LIB_UTIL_H_

//#include "Logger.h"

#include "llvm/Demangle/Demangle.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"

namespace typeart {
namespace util {

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
inline String demangle(String s) {
  //  static char buffer[1024];
  //  size_t n = 1024;
  return String(llvm::itaniumDemangle(s.data(), nullptr, nullptr, nullptr));
}

inline bool regex_matches(const std::string& regex, const std::string& in, bool case_sensitive = false) {
  using namespace llvm;
  Regex r(regex, !case_sensitive ? Regex::IgnoreCase : Regex::NoFlags);
  return r.match(in);
}

}  // namespace util
}  // namespace typeart

#endif /* LIB_UTIL_H_ */

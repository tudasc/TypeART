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
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"

namespace typeart {
namespace util {

#define ifcast(ty, var, val) if (ty* var = llvm::dyn_cast<ty>(val))  // NOLINT

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
  std::string name = s;
  auto demangle    = llvm::itaniumDemangle(name.data(), nullptr, nullptr, nullptr);
  if (demangle && std::string(demangle) != "") {
    return std::string(demangle);
  }
  return name;
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
        if (strchr("()^$|*+.\\", c)) {
          glob_reg += '\\';
        }
        glob_reg += c;
        break;
    }
  }
  glob_reg += "$";
  return glob_reg;
}

inline llvm::DILocalVariable* getDebugVar(const llvm::Value& inst, const llvm::Function& f) {
  llvm::DILocalVariable* var = nullptr;
  for (auto it = inst_begin(f); it != inst_end(f); it++) {
    ifcast(const llvm::DbgDeclareInst, dbgInst, &*it) {
      if (dbgInst->getAddress() == &inst) {
        var = dbgInst->getVariable();
        break;
      }
    }
    ifcast(const llvm::DbgValueInst, dbgInst, &*it) {
      if (dbgInst->getValue() == &inst) {
        var = dbgInst->getVariable();
        break;
      }
    }
  }

  return var;
}

inline llvm::DILocalVariable* getDebugVar(const llvm::Instruction& inst) {
  using namespace llvm;
  const llvm::Function& f = *inst.getFunction();

  ifcast(const llvm::CallInst, heapv, &inst) {
    // LOG_FATAL(util::dump(*heapv));
    for (auto user : heapv->users()) {
      if (auto storeInst = dyn_cast<StoreInst>(user)) {
        // LOG_FATAL(util::dump(*storeInst->getPointerOperand()));
        return getDebugVar(*storeInst->getPointerOperand(), f);
      }
    }
  }

  return getDebugVar(inst, f);
}

}  // namespace util
}  // namespace typeart

#endif /* LIB_UTIL_H_ */

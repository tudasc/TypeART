#ifndef TYPEART_LIB_PASSES_COMPAT_STLEXTRAS_H
#define TYPEART_LIB_PASSES_COMPAT_STLEXTRAS_H

#if LLVM_VERSION_MAJOR < 11
#include <type_traits>

namespace llvm {
namespace detail {
template <class, template <class...> class Op, class... Args>
struct detector {
  using value_t = std::false_type;
};
template <template <class...> class Op, class... Args>
struct detector<std::void_t<Op<Args...>>, Op, Args...> {
  using value_t = std::true_type;
};
}  // end namespace detail

/// Detects if a given trait holds for some set of arguments 'Args'.
/// For example, the given trait could be used to detect if a given type
/// has a copy assignment operator:
///   template<class T>
///   using has_copy_assign_t = decltype(std::declval<T&>()
///                                                 = std::declval<const T&>());
///   bool fooHasCopyAssign = is_detected<has_copy_assign_t, FooClass>::value;
template <template <class...> class Op, class... Args>
using is_detected = typename detail::detector<void, Op, Args...>::value_t;
}  // namespace llvm
#endif
#endif  // TYPEART_LIB_PASSES_COMPAT_STLEXTRAS_H

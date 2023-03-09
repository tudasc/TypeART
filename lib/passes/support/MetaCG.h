// TypeART library
//
// Copyright (c) 2017-2023 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#ifndef TYPEART_METACG_H
#define TYPEART_METACG_H

#include <llvm/ADT/StringMap.h>
#include <llvm/Support/JSON.h>
#include <string>
#include <vector>

using namespace llvm;

namespace typeart::util::metacg {

template <typename>
struct Extension;

template <typename...>
struct Node;

template <>
struct Node<> {
  std::vector<std::string> callees;
  std::vector<std::string> callers;
  std::vector<std::string> overrides;
  std::vector<std::string> overriddenBy;

  bool doesOverride = false;
  bool hasBody      = false;
  bool isVirtual    = false;
};

template <typename... Mixins>
struct Meta : public Extension<Mixins>... {};

template <typename... Extensions>
struct Node : public Node<> {
  Meta<Extensions...> meta;
};

struct Version {
  unsigned major{};
  unsigned minor{};
};

struct Generator {
  Version version{};
  std::string name{};
};

struct Info {
  Version version{};
  Generator generator{};
};

template <typename... Extensions>
struct MetaCG {
  using node_type = Node<Extensions...>;

  Info info;
  llvm::StringMap<node_type> nodes;
};

#if LLVM_VERSION_MAJOR < 12
template <typename T, typename A>
inline bool fromJSON(const json::Value& E, StringMap<T, A>& Out) {
  if (const auto* O = E.getAsObject()) {
    Out.clear();
    for (const auto& [K, V] : *O) {
      if (!fromJSON(V, Out[K])) {
        return false;
      }
    }
    return true;
  }
  return false;
}
#else
template <typename T, typename A>
inline bool fromJSON(const json::Value& E, StringMap<T, A>& Out, json::Path P) {
  if (const auto* O = E.getAsObject()) {
    Out.clear();
    for (const auto& [K, V] : *O) {
      if (!fromJSON(V, Out[K], P.field(K))) {
        return false;
      }
    }
    return true;
  }
  P.report("expected object");
  return false;
}
#endif

#if LLVM_VERSION_MAJOR < 12
template <typename T>
inline bool fromJSON(const json::Value& E, SmallVectorImpl<T>& Out) {
  if (const auto* A = E.getAsArray()) {
    Out.clear();
    Out.resize(A->size());
    for (size_t I = 0; I < A->size(); ++I) {
      if (!fromJSON((*A)[I], Out[I])) {
        return false;
      }
    }
    return true;
  }
  return false;
}
#else
template <typename T>
inline bool fromJSON(const json::Value& E, SmallVectorImpl<T>& Out, json::Path P) {
  if (const auto* A = E.getAsArray()) {
    Out.clear();
    Out.resize(A->size());
    for (size_t I = 0; I < A->size(); ++I) {
      if (!fromJSON((*A)[I], Out[I], P.index(I))) {
        return false;
      }
    }
    return true;
  }
  P.report("expected array");
  return false;
}
#endif

#if LLVM_VERSION_MAJOR < 12
inline bool fromJSON(const json::Value& E, Version& R) {
  auto S = E.getAsString();
  if (!S) {
    return false;
  }

  const auto& [MajorVer, MinorVer] = S->split('.');
  return !MajorVer.getAsInteger(10, R.major) && !MinorVer.getAsInteger(10, R.minor);
}
#else
inline bool fromJSON(const json::Value& E, Version& R, json::Path P) {
  auto S = E.getAsString();
  if (!S) {
    P.report("expected string");
    return false;
  }

  const auto& [MajorVer, MinorVer] = S->split('.');
  if (MajorVer.getAsInteger(10, R.major) || MinorVer.getAsInteger(10, R.minor)) {
    P.report("invalid version-format");
    return false;
  }

  return true;
}
#endif

#if LLVM_VERSION_MAJOR < 12
template <typename... Extensions>
inline bool fromJSON(const json::Value& E, MetaCG<Extensions...>& R) {
  json::ObjectMapper O(E);
  return O && O.map("_MetaCG", R.info) && O.map("_CG", R.nodes);
}
#else
template <typename... Extensions>
inline bool fromJSON(const json::Value& E, MetaCG<Extensions...>& R, json::Path P) {
  json::ObjectMapper O(E, P);

  if (O && O.map("_MetaCG", R.info)) {
    return O.map("_CG", R.nodes);
  }

  P.report("only MetaCG version 2.x format is supported");
  return false;
}
#endif

template <typename... Extra>
inline bool fromJSON(const json::Value& E, Info& R, Extra... P) {
  json::ObjectMapper O(E, P...);
  return O && O.map("version", R.version) && O.map("generator", R.generator);
}

template <typename... Extra>
inline bool fromJSON(const json::Value& E, Generator& R, Extra... P) {
  json::ObjectMapper O(E, P...);
  return O && O.map("version", R.version) && O.map("name", R.name);
}

template <typename... Extensions, typename... Extra>
inline bool fromJSON(const json::Value& E, Node<Extensions...>& R, Extra... P) {
  constexpr const bool HasExtensions = sizeof...(Extensions) > 0;

  json::ObjectMapper O(E, P...);

  if (O && O.map("hasBody", R.hasBody) && O.map("doesOverride", R.doesOverride) && O.map("isVirtual", R.isVirtual) &&
      O.map("overrides", R.overrides) && O.map("overriddenBy", R.overriddenBy) && O.map("callees", R.callees) &&
      O.map("callers", R.callers)) {
    if constexpr (HasExtensions) {
      return O.map("meta", R.meta);
    }
    return true;
  }
  return false;
}

template <typename... Extensions, typename... Extra>
inline bool fromJSON(const json::Value& E, Meta<Extensions...>& R, Extra... P) {
  // as the type "Meta" is an aggregation of extensions, we need to upcast
  // Meta to every one of its base classes and call the specific fromJSON variant
  return (fromJSON(E, static_cast<Extension<Extensions>&>(R), P...) && ...);
}

}  // namespace typeart::util::metacg
#endif  // TYPEART_METACG_H

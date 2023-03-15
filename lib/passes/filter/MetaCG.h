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

#ifndef TYPEART_FILTER_METACG_H
#define TYPEART_FILTER_METACG_H

#include <llvm/ADT/StringMap.h>
#include <llvm/Support/JSON.h>
#include <string>
#include <vector>

using namespace llvm;

namespace typeart::filter::metacg {

template <typename>
struct FunctionNode;

/// represents a function node without an additional meta field
template <>
struct FunctionNode<void> {
  std::vector<std::string> callees;
  std::vector<std::string> callers;
  std::vector<std::string> overrides;
  std::vector<std::string> overridden_by;

  bool does_override = false;
  bool has_body      = false;
  bool is_virtual    = false;
};

/// a function node with meta fields is basically an extension of a normal function node
template <typename MetaField>
struct FunctionNode : public FunctionNode<void> {
  MetaField meta;
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

template <typename MetaField = void>
struct MetaCG {
  using node_type = FunctionNode<MetaField>;

  Info info;
  llvm::StringMap<node_type> function_nodes;
};

// the fromJSON signature was changed with llvm12 so, we need support both variants - the old and the new one

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
template <typename MetaField>
inline bool fromJSON(const json::Value& E, MetaCG<MetaField>& R) {
  json::ObjectMapper O(E);
  return O && O.map("_MetaCG", R.info) && O.map("_CG", R.function_nodes);
}
#else
template <typename MetaField>
inline bool fromJSON(const json::Value& E, MetaCG<MetaField>& R, json::Path P) {
  json::ObjectMapper O(E, P);

  if (O && O.map("_MetaCG", R.info)) {
    return O.map("_CG", R.function_nodes);
  }

  P.report("only MetaCG version 2.x format is supported");
  return false;
}
#endif

#if LLVM_VERSION_MAJOR < 12
inline bool fromJSON(const json::Value& E, Info& R) {
  json::ObjectMapper O(E);
  return O && O.map("version", R.version) && O.map("generator", R.generator);
}
#else
inline bool fromJSON(const json::Value& E, Info& R, json::Path P) {
  json::ObjectMapper O(E, P);
  return O && O.map("version", R.version) && O.map("generator", R.generator);
}
#endif

#if LLVM_VERSION_MAJOR < 12
inline bool fromJSON(const json::Value& E, Generator& R) {
  json::ObjectMapper O(E);
  return O && O.map("version", R.version) && O.map("name", R.name);
}
#else

inline bool fromJSON(const json::Value& E, Generator& R, json::Path P) {
  json::ObjectMapper O(E, P);
  return O && O.map("version", R.version) && O.map("name", R.name);
}
#endif

#if LLVM_VERSION_MAJOR < 12
template <typename MetaField>
inline bool fromJSON(const json::Value& E, FunctionNode<MetaField>& R) {
  json::ObjectMapper O(E);
  return fromJSON<void>(E, R) && O && O.map("meta", R.meta);
}

template <>
inline bool fromJSON(const json::Value& E, FunctionNode<void>& R) {
  json::ObjectMapper O(E);
  return O && O.map("hasBody", R.has_body) && O.map("doesOverride", R.does_override) &&
         O.map("isVirtual", R.is_virtual) && O.map("overrides", R.overrides) &&
         O.map("overriddenBy", R.overridden_by) && O.map("callees", R.callees) && O.map("callers", R.callers);
}
#else
template <typename MetaField>
inline bool fromJSON(const json::Value& E, FunctionNode<MetaField>& R, json::Path P) {
  json::ObjectMapper O(E, P);
  return fromJSON<void>(E, R, P) && O && O.map("meta", R.meta);
}

template <>
inline bool fromJSON(const json::Value& E, FunctionNode<void>& R, json::Path P) {
  json::ObjectMapper O(E, P);
  return O && O.map("hasBody", R.has_body) && O.map("doesOverride", R.does_override) &&
         O.map("isVirtual", R.is_virtual) && O.map("overrides", R.overrides) &&
         O.map("overriddenBy", R.overridden_by) && O.map("callees", R.callees) && O.map("callers", R.callers);
}
#endif

}  // namespace typeart::filter::metacg
#endif  // TYPEART_FILTER_METACG_H

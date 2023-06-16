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

#ifndef TYPEART_FILTER_METACG_EXTENSION_H
#define TYPEART_FILTER_METACG_EXTENSION_H

#include "MetaCG.h"

#include <llvm/Support/JSON.h>
#include <string>
#include <vector>

using namespace llvm;

namespace typeart::filter::metacg {

template <typename>
struct MetaField;

#if LLVM_VERSION_MAJOR < 12
template <typename Extension>
inline bool fromJSON(const json::Value& E, MetaField<Extension>& R);
#else
template <typename Extension>
inline bool fromJSON(const json::Value& E, MetaField<Extension>& R, json::Path P);
#endif

struct FunctionSignature {
  std::string identifier;
  std::vector<std::string> param_types;
  std::string return_type;
  bool is_variadic = false;
};

template <>
struct MetaField<FunctionSignature> {
  FunctionSignature signature;
};

#if LLVM_VERSION_MAJOR < 12
template <>
inline bool fromJSON(const json::Value& E, MetaField<FunctionSignature>& R) {
  json::ObjectMapper O(E);
  return O && O.map("signature", R.signature);
}
#else
template <>
inline bool fromJSON(const json::Value& E, MetaField<FunctionSignature>& R, json::Path P) {
  json::ObjectMapper O(E, P);
  return O && O.map("signature", R.signature);
}
#endif

#if LLVM_VERSION_MAJOR < 12
inline bool fromJSON(const json::Value& E, FunctionSignature& R) {
  json::ObjectMapper O(E);
  return O && O.map("identifier", R.identifier) && O.map("paramTypes", R.param_types) &&
         O.map("isVariadic", R.is_variadic) && O.map("returnType", R.return_type);
}
#else
inline bool fromJSON(const json::Value& E, FunctionSignature& R, json::Path P) {
  json::ObjectMapper O(E, P);
  return O && O.map("identifier", R.identifier) && O.map("paramTypes", R.param_types) &&
         O.map("isVariadic", R.is_variadic) && O.map("returnType", R.return_type);
}
#endif

struct InterDataFlow {
  struct CallSite {
    int64_t site_identifier{};
  };

  struct Edge {
    /// position number of the source function
    int source_argument_number{};

    /// position number of the destination function
    int sink_argument_number{};
  };

  llvm::StringMap<std::vector<CallSite>> callsites{};
  llvm::StringMap<std::vector<Edge>> inter_flow{};
};

template <>
struct MetaField<InterDataFlow> {
  InterDataFlow ipdf;
};

#if LLVM_VERSION_MAJOR < 12
template <>
inline bool fromJSON(const json::Value& E, MetaField<InterDataFlow>& R) {
  json::ObjectMapper O(E);
  return O && O.map("ipdf", R.ipdf);
}
#else
template <>
inline bool fromJSON(const json::Value& E, MetaField<InterDataFlow>& R, json::Path P) {
  json::ObjectMapper O(E, P);
  return O && O.map("ipdf", R.ipdf);
}
#endif

#if LLVM_VERSION_MAJOR < 12
inline bool fromJSON(const json::Value& E, InterDataFlow& R) {
  json::ObjectMapper O(E);
  return O && O.map("interFlow", R.inter_flow) && O.map("callSites", R.callsites);
}
#else
inline bool fromJSON(const json::Value& E, InterDataFlow& R, json::Path P) {
  json::ObjectMapper O(E, P);
  return O && O.map("interFlow", R.inter_flow) && O.map("callSites", R.callsites);
}
#endif

#if LLVM_VERSION_MAJOR < 12
inline bool fromJSON(const json::Value& E, InterDataFlow::CallSite& R) {
  return fromJSON(E, R.site_identifier);
}
#else
inline bool fromJSON(const json::Value& E, InterDataFlow::CallSite& R, json::Path P) {
  return fromJSON(E, R.site_identifier, P);
}
#endif

#if LLVM_VERSION_MAJOR < 12
inline bool fromJSON(const json::Value& E, InterDataFlow::Edge& R) {
  json::ObjectMapper O(E);
  return O && O.map("sink", R.sink_argument_number) && O.map("source", R.source_argument_number);
}
#else
inline bool fromJSON(const json::Value& E, InterDataFlow::Edge& R, json::Path P) {
  json::ObjectMapper O(E, P);
  return O && O.map("sink", R.sink_argument_number) && O.map("source", R.source_argument_number);
}
#endif

/// an aggregation to allow the usage of multiple extensions
template <typename... Extensions>
struct MetaFieldGroup : public MetaField<Extensions>... {
 private:
#if LLVM_VERSION_MAJOR < 12
  friend inline bool fromJSON(const json::Value& E, MetaFieldGroup& R) {
    // call every fromJSON specialization for every super MetaField, this is possible as every
    // extension itself is a template specialization.
    return (fromJSON<Extensions>(E, R) && ...);
  }
#else
  friend inline bool fromJSON(const json::Value& E, MetaFieldGroup& R, json::Path P) {
    // call every fromJSON specialization for every super MetaField, this is possible as every
    // extension itself is a template specialization.
    return (fromJSON<Extensions>(E, R, P) && ...);
  }
#endif
};

}  // namespace typeart::filter::metacg
#endif  // TYPEART_FILTER_METACG_EXTENSION_H

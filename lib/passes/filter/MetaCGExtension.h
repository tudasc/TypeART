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

struct FunctionSignature {
  std::string identifier;
  std::vector<std::string> paramTypes;
  std::string returnType;
  bool isVariadic = false;
};

template <>
struct MetaFieldExtension<FunctionSignature> {
  FunctionSignature signature;
};

// the fromJSON signature was changed with llvm12 so, we need support both variants - the old and the new one

#if LLVM_VERSION_MAJOR < 12
inline bool fromJSON(const json::Value& E, MetaFieldExtension<FunctionSignature>& R) {
  json::ObjectMapper O(E);
  return O && O.map("signature", R.signature);
}
#else
inline bool fromJSON(const json::Value& E, MetaFieldExtension<FunctionSignature>& R, json::Path P) {
  json::ObjectMapper O(E, P);
  return O && O.map("signature", R.signature);
}
#endif

#if LLVM_VERSION_MAJOR < 12
inline bool fromJSON(const json::Value& E, FunctionSignature& R) {
  json::ObjectMapper O(E);
  return O && O.map("identifier", R.identifier) && O.map("paramTypes", R.paramTypes) &&
         O.map("isVariadic", R.isVariadic) && O.map("returnType", R.returnType);
}
#else
inline bool fromJSON(const json::Value& E, FunctionSignature& R, json::Path P) {
  json::ObjectMapper O(E, P);
  return O && O.map("identifier", R.identifier) && O.map("paramTypes", R.paramTypes) &&
         O.map("isVariadic", R.isVariadic) && O.map("returnType", R.returnType);
}
#endif

struct InterDataFlow {
  struct CallSite {
    int64_t id{};
  };

  struct Edge {
    /// position number of the source function
    int sourceArgumentNumber{};

    /// position number of the destination function
    int sinkArgumentNumber{};
  };

  llvm::StringMap<std::vector<CallSite>> callsites{};
  llvm::StringMap<std::vector<Edge>> interFlow{};
};

template <>
struct MetaFieldExtension<InterDataFlow> {
  InterDataFlow ipdf;
};

#if LLVM_VERSION_MAJOR < 12
inline bool fromJSON(const json::Value& E, MetaFieldExtension<InterDataFlow>& R) {
  json::ObjectMapper O(E);
  return O && O.map("ipdf", R.ipdf);
}
#else
inline bool fromJSON(const json::Value& E, MetaFieldExtension<InterDataFlow>& R, json::Path P) {
  json::ObjectMapper O(E, P);
  return O && O.map("ipdf", R.ipdf);
}
#endif

#if LLVM_VERSION_MAJOR < 12
inline bool fromJSON(const json::Value& E, InterDataFlow& R) {
  json::ObjectMapper O(E);
  return O && O.map("interFlow", R.interFlow) && O.map("callSites", R.callsites);
}
#else
inline bool fromJSON(const json::Value& E, InterDataFlow& R, json::Path P) {
  json::ObjectMapper O(E, P);
  return O && O.map("interFlow", R.interFlow) && O.map("callSites", R.callsites);
}
#endif

#if LLVM_VERSION_MAJOR < 12
inline bool fromJSON(const json::Value& E, InterDataFlow::CallSite& R) {
  return fromJSON(E, R.id);
}
#else
inline bool fromJSON(const json::Value& E, InterDataFlow::CallSite& R, json::Path P) {
  return fromJSON(E, R.id, P);
}
#endif

#if LLVM_VERSION_MAJOR < 12
inline bool fromJSON(const json::Value& E, InterDataFlow::Edge& R) {
  json::ObjectMapper O(E);
  return O && O.map("sink", R.sinkArgumentNumber) && O.map("source", R.sourceArgumentNumber);
}
#else
inline bool fromJSON(const json::Value& E, InterDataFlow::Edge& R, json::Path P) {
  json::ObjectMapper O(E, P);
  return O && O.map("sink", R.sinkArgumentNumber) && O.map("source", R.sourceArgumentNumber);
}
#endif

}  // namespace typeart::filter::metacg
#endif  // TYPEART_FILTER_METACG_EXTENSION_H

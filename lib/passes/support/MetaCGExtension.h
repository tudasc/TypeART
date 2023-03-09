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

#ifndef TYPEART_METACG_EXTENSION_H
#define TYPEART_METACG_EXTENSION_H

#include "MetaCG.h"

#include <llvm/Support/JSON.h>
#include <string>
#include <vector>

using namespace llvm;

namespace typeart::util::metacg {

struct Signature {
  std::string identifier;
  std::vector<std::string> paramTypes;
  std::string returnType;
  bool isVariadic = false;
};

template <>
struct Extension<Signature> {
  Signature signature;
};

// the fromJSON signature was changed with llvm12 so, we need support both variants - the old and the new one

template <typename... Extra>
inline bool fromJSON(const json::Value& E, Extension<Signature>& R, Extra... P) {
  json::ObjectMapper O(E, P...);
  return O && O.map("signature", R.signature);
}

template <typename... Extra>
inline bool fromJSON(const json::Value& E, Signature& R, Extra... P) {
  json::ObjectMapper O(E, P...);
  return O && O.map("identifier", R.identifier) && O.map("paramTypes", R.paramTypes) &&
         O.map("isVariadic", R.isVariadic) && O.map("returnType", R.returnType);
}

struct InterDataFlow {
  struct CallSite {
    int64_t id{};
  };

  struct Edge {
    /// position number of the source function
    int actualArgNo{};

    /// position number of the destination function
    int formalArgNo{};
  };

  llvm::StringMap<std::vector<CallSite>> callsites{};
  llvm::StringMap<std::vector<Edge>> interFlow{};
};

template <>
struct Extension<InterDataFlow> {
  InterDataFlow ipdf;
};

template <typename... Extra>
inline bool fromJSON(const json::Value& E, Extension<InterDataFlow>& R, Extra... P) {
  json::ObjectMapper O(E, P...);
  return O && O.map("ipdf", R.ipdf);
}

template <typename... Extra>
inline bool fromJSON(const json::Value& E, InterDataFlow& R, Extra... P) {
  json::ObjectMapper O(E, P...);
  return O && O.map("interFlow", R.interFlow) && O.map("callSites", R.callsites);
}

template <typename... Extra>
inline bool fromJSON(const json::Value& E, InterDataFlow::CallSite& R, Extra... P) {
  return fromJSON(E, R.id, P...);
}

template <typename... Extra>
inline bool fromJSON(const json::Value& E, InterDataFlow::Edge& R, Extra... P) {
  json::ObjectMapper O(E, P...);
  return O && O.map("formal", R.formalArgNo) && O.map("actual", R.actualArgNo);
}

}  // namespace typeart::util::metacg
#endif  // TYPEART_METACG_EXTENSION_H

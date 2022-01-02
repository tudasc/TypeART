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

#ifndef TYPEART_IRPATH_H
#define TYPEART_IRPATH_H

#include "compat/CallSite.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/raw_ostream.h"

#include <unordered_map>
#include <vector>

namespace typeart::filter {

struct IRPath {
  using Node = llvm::Value*;
  std::vector<llvm::Value*> path;
  // FIXME
  //  this mechanism tries to avoid endless recursion in loops, i.e.,
  //  do we bounce around multiple phi nodes (visit counter >1), then
  //  we should likely skip search, see IRSearch.h
  std::unordered_map<llvm::Value*, int> phi_cache;

  llvm::Optional<Node> getStart() const {
    if (path.empty()) {
      return llvm::None;
    }
    return *path.begin();
  }

  llvm::Optional<Node> getEnd() const {
    return getNodeFromEnd<1>();
  }

  llvm::Optional<Node> getEndPrev() const {
    return getNodeFromEnd<2>();
  }

  template <unsigned n>
  llvm::Optional<Node> getNodeFromEnd() const {
    if (path.empty() || path.size() < n) {
      return llvm::None;
    }
    return *std::prev(path.end(), n);
  }

  void pop() {
    if (!path.empty()) {
      path.pop_back();
    }
  }

  void push(Node n) {
    if (llvm::isa<llvm::PHINode>(n)) {
      ++(phi_cache[n]);
    }
    path.push_back(n);
  }

  bool contains(Node n) const {
    return llvm::find_if(path, [&n](const auto* node) { return node == n; }) != std::end(path);
  }
};

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const IRPath& p) {
  const auto& vec = p.path;
  if (vec.empty()) {
    os << "path = [ ]";
    return os;
  }
  auto begin = std::begin(vec);
  os << "path = [" << **begin;
  std::for_each(std::next(begin), std::end(vec), [&](const auto* v) {
    os << " ->";
    if (const auto f = llvm::dyn_cast<llvm::Function>(v); f != nullptr && !f->isDeclaration()) {
      // do not print body of defined function, from "define" to "{"
      std::string buf;
      llvm::raw_string_ostream function_ostream(buf);
      function_ostream << *f;

      llvm::StringRef fref(function_ostream.str());
      auto pos_start     = fref.find("define");
      const auto pos_end = fref.find("{");
      if (pos_start == llvm::StringRef::npos || pos_start > pos_end) {
        pos_start = 0;
      }
      auto fsig = fref.substr(pos_start, pos_end - pos_start);
      os << " Function: " << fsig;
    } else {
      os << *v;
    }
  });
  os << "]";
  return os;
}

struct CallsitePath {
  // Node: IRPath leads to function:
  using Node = std::pair<llvm::Function*, IRPath>;

  // Structure: [Function (start) or null for global]{1} -> [Path -> Function]* -> (Path)?
  llvm::Optional<llvm::Function*> start;
  IRPath terminatingPath{};
  std::vector<Node> intermediatePath;

  explicit CallsitePath(llvm::Function* root) {
    if (root == nullptr) {
      start = llvm::None;
    } else {
      start = root;
    }
  }

  bool empty() const {
    return intermediatePath.empty() && terminatingPath.path.empty();
  }

  // Can return nullptr
  llvm::Function* getCurrentFunc() {
    if (intermediatePath.empty()) {
      if (start) {
        return start.getValue();
      }
      return nullptr;
    }
    auto end = getEnd();
    if (end) {
      return end.getValue().first;
    }
    return nullptr;
  }

  llvm::Optional<Node> getStart() const {
    if (intermediatePath.empty()) {
      return llvm::None;
    }
    return *intermediatePath.begin();
  }

  llvm::Optional<Node> getEnd() const {
    return getNodeFromEnd<1>();
  }

  template <unsigned n>
  llvm::Optional<Node> getNodeFromEnd() const {
    if (intermediatePath.empty() || intermediatePath.size() < n) {
      return llvm::None;
    }
    return *std::prev(intermediatePath.end(), n);
  }

  void push(const IRPath& p) {
    auto csite = p.getEnd();
    if (csite) {
      // Omp extension: we may pass the outlined area directly as llvm::Function
      if (auto f = llvm::dyn_cast<llvm::Function>(csite.getValue())) {
        intermediatePath.emplace_back(f, p);
        return;
      }
      llvm::CallSite c(csite.getValue());
      intermediatePath.emplace_back(c.getCalledFunction(), p);
    }
  }

  void pushFinal(const IRPath& p) {
    terminatingPath = p;
  }

  void pop() {
    if (!intermediatePath.empty()) {
      intermediatePath.pop_back();
    }
  }

  bool contains(llvm::CallSite c) {
    llvm::Function* f = c.getCalledFunction();
    if ((f != nullptr) && f == start) {
      return true;
    }
    return llvm::find_if(intermediatePath, [&f](const auto& node) { return node.first == f; }) !=
           std::end(intermediatePath);
  }
};

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const CallsitePath::Node& n) {
  auto f = n.first;
  if (f != nullptr) {
    os << f->getName();
  } else {
    os << "--";
  }
  os << ":" << n.second;
  return os;
}

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const CallsitePath& p) {
  const auto& vec = p.intermediatePath;
  if (vec.empty()) {
    os << "func_path = [";
    if (p.start) {
      os << p.start.getValue()->getName();
    } else {
      os << "Module";
    }
    os << " -> " << p.terminatingPath << "]";
    return os;
  }
  auto begin = std::begin(vec);
  os << "func_path = [" << *begin;
  std::for_each(std::next(begin), std::end(vec), [&](const auto& v) { os << " -> " << v; });
  os << "]";
  return os;
}

using Path  = IRPath;
using FPath = CallsitePath;

using PathList = std::vector<Path>;

}  // namespace typeart::filter

#endif  // TYPEART_IRPATH_H

//
// Created by ahueck on 03.11.20.
//

#ifndef TYPEART_IRPATH_H
#define TYPEART_IRPATH_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/raw_ostream.h"

#include <vector>

struct IRPath {
  using Node = llvm::Value*;
  std::vector<llvm::Value*> path;

  llvm::Optional<Node> bottom() const {
    if (path.empty()) {
      return llvm::None;
    }
    return *path.begin();
  }

  llvm::Optional<Node> top() const {
    return topPos<1>();
  }

  llvm::Optional<Node> top2nd() const {
    return topPos<2>();
  }

  template <unsigned n>
  llvm::Optional<Node> topPos() const {
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
  std::for_each(std::next(begin), std::end(vec), [&](const auto* v) { os << " ->" << *v; });
  os << "]";
  return os;
}

using Path = IRPath;

#endif  // TYPEART_IRPATH_H

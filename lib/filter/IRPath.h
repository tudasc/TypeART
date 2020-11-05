//
// Created by ahueck on 03.11.20.
//

#ifndef TYPEART_IRPATH_H
#define TYPEART_IRPATH_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/raw_ostream.h"

#include <vector>

struct IRPath {
  using Node = llvm::Value*;
  std::vector<llvm::Value*> path;

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

struct CallsitePath {
  using Node = std::pair<llvm::Function*, IRPath>;
  std::vector<Node> path;

  explicit CallsitePath(llvm::Function* root) {
    path.emplace_back(root, IRPath{});
  }

  llvm::Optional<llvm::Function*> getCurrentFunc() {
    if (path.empty()) {
      return llvm::None;
    }
    auto end = getEnd();
    return end.getValue().first;
  }

  llvm::Optional<Node> getStart() const {
    if (path.empty()) {
      return llvm::None;
    }
    return *path.begin();
  }

  llvm::Optional<Node> getEnd() const {
    return getNodeFromEnd<1>();
  }

  template <unsigned n>
  llvm::Optional<Node> getNodeFromEnd() const {
    if (path.empty() || path.size() < n) {
      return llvm::None;
    }
    return *std::prev(path.end(), n);
  }

  void push(const IRPath& p) {
    auto csite = p.getEnd();
    if (csite) {
      llvm::CallSite c(csite.getValue());
      path.emplace_back(c.getCalledFunction(), p);
    }
  }

  void pop() {
    if (!path.empty()) {
      path.pop_back();
    }
  }

  bool contains(llvm::CallSite c) {
    llvm::Function* f = c.getCalledFunction();
    return llvm::find_if(path, [&f](const auto& node) { return node.first == f; }) != std::end(path);
  }
};

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const CallsitePath::Node& n) {
  os << n.first->getName() << ":" << n.second;
  return os;
}

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const CallsitePath& p) {
  const auto& vec = p.path;
  if (vec.empty()) {
    os << "func_path = [ ]";
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

#endif  // TYPEART_IRPATH_H

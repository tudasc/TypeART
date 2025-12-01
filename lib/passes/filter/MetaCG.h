// TypeART library
//
// Copyright (c) 2017-2025 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#ifndef METACG_H
#define METACG_H

#include <llvm/Support/JSON.h>

#include <support/Logger.h>

#include <string_view>
#include <charconv>
#include <utility>

template <>
struct std::hash<std::pair<size_t, size_t>> {
  size_t operator()(std::pair<size_t, size_t> const& p) const noexcept {
    size_t h1 = std::hash<size_t>{}(std::get<0>(p));
    size_t h2 = std::hash<size_t>{}(std::get<1>(p));
    return h1 ^ h2 << 1;
  }
};

namespace typeart::filter::metacg {

/// Holds information about the generator used to serialize the callgraph
struct Generator { std::string name, sha, version; };

namespace json = llvm::json;

inline bool fromJSON(json::Value const& json, Generator& r, json::Path const& path) {
  json::ObjectMapper o{json, path};
  return o && o.map("name", r.name) && o.map("sha", r.sha) && o.map("version", r.version);
}

/// MetaCGv4 header
struct Header {
  Generator gen;
  std::string version;
};

inline bool fromJSON(json::Value const& json, Header& r, json::Path const& path) {
  json::ObjectMapper o{json, path};
  return o && o.map("generator", r.gen) && o.map("version", r.version);
}

/// Represents a source location
struct SrcLoc {
  auto operator==(const SrcLoc& rhs) const { return col == rhs.col && line == rhs.line; }

  size_t col, line;
};

inline bool fromJSON(llvm::json::Value const& json, SrcLoc& r, json::Path const& path) {
  json::ObjectMapper o{json, path};
  return o && o.map("col", r.col) && o.map("line", r.line);
}

/// Call instances metadata
struct MdCalls { std::vector<SrcLoc> locs; };

inline bool fromJSON(json::Value const& json, MdCalls& r, json::Path const& path) {
  json::ObjectMapper o{json, path};
  return o && o.map("locs", r.locs);
}

/// Represents an output parameter an incoming argument flows into
struct MdArgOutput {
  bool by_ref;
  std::vector<size_t> callees;
  size_t idx;
  SrcLoc loc;
};

inline bool fromJSON(json::Value const& json, MdArgOutput& r, json::Path const& path) {
  json::ObjectMapper o{json, path};
  return o
    && o.map("by_ref", r.by_ref)
    && o.map("callees", r.callees)
    && o.map("idx", r.idx)
    && o.map("loc", r.loc);
}

/// Represents an incoming argument
struct MdArg {
  size_t idx;
  std::vector<MdArgOutput> outs;
};

inline bool fromJSON(json::Value const& json, MdArg& r, json::Path const& path) {
  json::ObjectMapper o{json, path};
  return o && o.map("idx", r.idx) && o.map("outs", r.outs);
}

/// Represents a local that flows into a callee
struct MdLocal {
  std::vector<size_t> callees;
  SrcLoc loc;
};

inline bool fromJSON(json::Value const& json, MdLocal& r, json::Path const& path) {
  json::ObjectMapper o{json, path};
  return o && o.map("callees", r.callees) && o.map("loc", r.loc);
}

/// Toplevel node metadata to annotate each published parameter
struct MdArgflow {
  std::vector<MdArg> args;
};

inline bool fromJSON(json::Value const& json, MdArgflow& r, json::Path const& path) {
  json::ObjectMapper o{json, path};
  return o && o.map("args", r.args);
}

/// Toplevel node metadata to annotate published locals
struct MdLocals {
  std::vector<MdLocal> locals;
};

inline bool fromJSON(json::Value const& json, MdLocals& r, json::Path const& path) {
  json::ObjectMapper o{json, path};
  return o && o.map("locals", r.locals);
}

/// Represents a generic metadata entry, required as metadata entries are inherently polymorphic
/// so the concrete deserialization is left to the caller.
struct Md {
  friend bool fromJSON(json::Value const&, Md&, json::Path const&);

  Md() = default;

  explicit Md(json::Value const& val) : v{val} {}

  /// Performs a lookup into the metadata object with the given key `name` and attempts to deserialize
  /// it into the requested type.
  template <typename T>
  llvm::Expected<T> as(llvm::StringRef const name) const {
    if (v.getAsNull())
      return llvm::createStringError("No metadata object");

    json::Path::Root root{};
    auto p = v.getAsObject()->get(name);
	 if (!p)
		 return llvm::createStringError("Member not found");

    if (T r{}; fromJSON(*p, r, root))
      return r;

    return llvm::createStringError("Failed to deserialize metadata");
  }

private:
  json::Value v{nullptr};
};

inline bool fromJSON(json::Value const& json, Md& r, json::Path const&) {
  r.v = json;
  return true;
}

/// Represents edges from a node to its callees and associated edge metadata
using Edges = std::unordered_map<size_t, Md>;

inline bool fromJSON(json::Value const& json, Edges& r, json::Path const& path) {
  auto const outer = json.getAsObject();
  if (!outer)
    return false;

  for (auto const& [k, v] : *outer) {
    auto k_str = k.str();

    size_t hash{};
    if (std::from_chars(k_str.data(), k_str.data() + k_str.size(), hash).ec == std::errc{})
      r.insert({hash, Md{v}});
    else
      return false;
  }

  return true;
}

/// Represents a node in the callgraph
struct Node {
  std::optional<std::string> name;
  std::optional<std::string> origin;
  bool has_body;
  Md meta;
  Edges callees;
};

inline bool fromJSON(json::Value const& json, Node& r, json::Path const& path) {
  json::ObjectMapper o{json, path};
  return o
    && o.map("functionName", r.name)
    && o.map("origin", r.origin)
    && o.map("hasBody", r.has_body)
    && o.map("meta", r.meta)
    && o.map("callees", r.callees);
}

/// Represents the callgraph's nodes and their associated IDs
using Nodes = std::unordered_map<size_t, Node>;

inline bool fromJSON(json::Value const& json, Nodes& r, json::Path const& path) {
  auto const outer = json.getAsObject();
  if (!outer)
    return false;

  for (auto const& [k, v] : *outer) {
    auto k_str = k.str();

    Node node {};
    if (auto const parsed = fromJSON(v, node, path); !parsed)
      return false;

    size_t hash{};
    if (std::from_chars(k_str.data(), k_str.data() + k_str.size(), hash).ec == std::errc{})
      r.insert({hash, node});
    else
      return false;
  }

  return true;
}

/// MetaCGv4 callgraph
struct CallGraph {
  Md meta;
  Nodes nodes;
};

inline bool fromJSON(json::Value const& json, CallGraph& r, json::Path const& path) {
  json::ObjectMapper o{json, path};
  return o
    && o.map("meta", r.meta)
    && o.map("nodes", r.nodes);
}

/// Top-level MetaCGv4 object container
struct Mcg {
  std::optional<size_t> byName(std::string_view const name) const {
    for (auto const& [hash, node] : this->graph.nodes)
      if (node.name == name)
        return hash;

    return {};
  }

  std::optional<Node> forId(size_t id) const {
    for (auto const& [hash, node] : this->graph.nodes)
      if (hash == id)
        return node;

    return {};
  }

  std::optional<std::vector<MdArgOutput>> outputs(size_t node, size_t idx) const {
    if (this->graph.nodes.find(node) == this->graph.nodes.end())
      return {};

    auto argflow = this->graph.nodes.at(node).meta.as<MdArgflow>("argflow");
    if (!argflow)
      return {};

    for (auto const& [id, out]: argflow->args) {
      if (id == idx)
        return out;
    }

    return {};
  }

  CallGraph graph;
  Header hdr;
};

inline bool fromJSON(json::Value const& json, Mcg& r,json::Path const& path) {
  json::ObjectMapper o{json, path};
  return o && o.map("_CG", r.graph) && o.map("_MetaCG", r.hdr);
}

/// Deserializes the MetaCGv4 format from text
inline llvm::Expected<Mcg> parse(llvm::StringRef const json) {
  if (auto parsed = json::parse<Mcg>(json); parsed) {
    LOG_DEBUG("Generated by: " << parsed->hdr.gen.name << " version: " << parsed->hdr.version); 
    return *parsed;
  }
  else
    return llvm::Expected<Mcg>{parsed.takeError()};
}

} // namespace typeart::filter::metacg

#endif //METACG_H

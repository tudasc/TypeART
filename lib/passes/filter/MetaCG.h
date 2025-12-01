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

inline bool fromJSON(json::Value const& json, Generator& r, json::Path const& path) {
  json::ObjectMapper o{json, path};
  return o && o.map("name", r.name) && o.map("sha", r.sha) && o.map("version", r.version);
}

/// MetaCGv3 header
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
  Expected<T> as(StringRef const name) const {
    if (v.getAsNull())
      return createStringError("No metadata object");

    json::Path::Root root{};
    if (T r{}; fromJSON(*(v.getAsObject()->get(name)), r, root))
      return r;

    return createStringError("Failed to deserialize metadata");
  }

private:
  json::Value v{nullptr};
};

inline bool fromJSON(json::Value const& json, Md& r, json::Path const&) {
  r.v = json;
  return true;
}

/// Represents a function in the callgraph
struct Fn {
  std::string name, origin;
  bool has_body;
  Md meta;
};

inline bool fromJSON(json::Value const& json, Fn& r, json::Path const& path) {
  json::ObjectMapper o{json, path};
  return o
    && o.map("functionName", r.name)
    && o.map("origin", r.origin)
    && o.map("hasBody", r.has_body)
    && o.map("meta", r.meta);
}

/// Represents the edges of the callgraph
struct EdgeContainer {
  friend struct CallGraph;

  friend bool fromJSON(json::Value const& json, EdgeContainer& r, json::Path const& path);

  EdgeContainer() = default;

private:
  std::unordered_map<std::pair<size_t, size_t>, Md> underlying{};
};

inline bool fromJSON(json::Value const& json, EdgeContainer& r, json::Path const& path) {
  auto const outer = json.getAsArray();
  if (!outer)
    return false;

  for (json::Value const& v: *outer) {
    auto const inner = v.getAsArray();
    if (!inner)
      return false;

    auto const first = (*inner)[0].getAsArray();
    auto const second = (*inner)[1];
    if (!first)
      return false;

    r.underlying.insert({
      {*(*first)[0].getAsUINT64(), *(*first)[1].getAsUINT64()},
      Md{second}
    });
  }

  return true;
}

/// Represents the nodes of the callgraph
struct NodeContainer {
  friend struct CallGraph;

  NodeContainer() = default;

  friend bool fromJSON(json::Value const& json, NodeContainer& r, json::Path const& path);

  /// Translates a function name to its corresponding node ID if it exists
  std::optional<size_t> byName(StringRef const name) const {
    for (auto const& [id, f]: underlying) {
      if (f.name == name)
        return id;
    }

    return {};
  }

  /// Translates a node identifier to a function descriptor
  std::optional<Fn> forId(size_t const id) const {
    for (auto const& [idx, f]: underlying) {
      if (idx == id)
        return f;
    }

    return {};
  }

  /// Returns the parameter outputs for `node`'s incoming argument at position `idx`
  std::optional<std::vector<MdArgOutput>> outputs(size_t node, size_t idx) const {
    if (underlying.find(node) == underlying.end())
      return {};

    auto argflow = underlying.at(node).meta.as<MdArgflow>("argflow");
    if (!argflow)
      return {};

    for (auto const& [id, out]: argflow->args) {
      if (id == idx)
        return out;
    }

    return {};
  }

private:
  std::unordered_map<size_t, Fn> underlying{};
};

inline bool fromJSON(json::Value const& json, NodeContainer& r, json::Path const& path) {
  auto const outer = json.getAsArray();
  if (!outer)
    return false;

  for (json::Value const& v: *outer) {
    auto const inner = v.getAsArray();
    if (!inner)
      return false;

    auto const first = (*inner)[0].getAsUINT64();
    if (!first)
      return false;

    Fn f{};
    if (auto const second = fromJSON((*inner)[1], f, path); !second)
      return false;

    r.underlying.insert({*first, f});
  }

  return true;
}

/// Represents the complete callgraph
struct CallGraph {
  EdgeContainer edges;
  NodeContainer nodes;
};

inline bool fromJSON(json::Value const& json, CallGraph& r, json::Path const& path) {
  json::ObjectMapper o{json, path};
  return o && o.map("edges", r.edges) && o.map("nodes", r.nodes);
}

/// Top-level MetaCGv3 object container
struct Mcg {
  CallGraph graph;
  Header hdr;
};

inline bool fromJSON(json::Value const& json, Mcg& r,json::Path const& path) {
  json::ObjectMapper o{json, path};
  return o && o.map("_CG", r.graph) && o.map("_MetaCG", r.hdr);
}

/// Deserializes the MetaCGv3 format from text
inline Expected<Mcg> parse(StringRef const json) {
  if (auto parsed = json::parse<Mcg>(json); parsed) {
    LOG_DEBUG("Generated by: " << parsed->hdr.gen.name << '\n' << "Version: " << parsed->hdr.version << '\n');
    return *parsed;
  }
  else
    return Expected<Mcg>{parsed.takeError()};
}

} // namespace typeart::filter::metacg

#endif //METACG_H

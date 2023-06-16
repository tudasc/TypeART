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

#include "ACGForwardFilter.h"

#include <llvm/IR/InstIterator.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/ModuleSummaryIndex.h>
#include <llvm/Support/SHA1.h>

using namespace llvm;

namespace typeart::filter {

namespace detail {

/// follows all enqueued values and visits them once starting with an initial entry

enum VisitResult {
  kContinue = 0,
  kStop,
};

template <typename T, typename CB>
inline void solveReachable(const std::vector<T>& range, CB&& callback) noexcept {
  llvm::SmallPtrSet<const T*, 32> visited{};
  llvm::SmallVector<const T*, 64> worklist{};

  const auto& enqueue = [&](const T& entry) noexcept -> bool {
    if (visited.insert(&entry).second) {
      worklist.push_back(&entry);
      return true;
    }
    return false;
  };

  for (const auto& item : range) {
    enqueue(item);
  }

  while (!worklist.empty()) {
    const T* current_value   = worklist.pop_back_val();
    const VisitResult status = callback(*current_value, enqueue);
    switch (status) {
      case kContinue:
        break;
      case kStop:
        return;
    }
  }
}

}  // namespace detail

static FunctionDescriptor createFunctionNode(const JsonACG::node_type& json, const llvm::Regex& target_matcher) {
  const auto& signature            = json.meta.signature;
  const bool is_target_of_interest = target_matcher.match(signature.identifier);

  return {is_target_of_interest,
          json.has_body,
          {},
          {},
          {signature.identifier, signature.param_types, signature.return_type, signature.is_variadic}};
}

static const FunctionDescriptor& retrieveFunction(const ACGDataMap& data_map, const llvm::StringRef& function_name) {
  assert(data_map.count(function_name) && "Found an edge to an undefined callee");
  if (data_map.count(function_name) == 0) {
    throw std::runtime_error("Found an edge to an undefined callee: " + function_name.str());
  }
  return data_map.find(function_name)->second;
}

static void insertCallsiteIdentifiers(const JsonACG::node_type& json, FunctionDescriptor& data, ACGDataMap& data_map) {
  llvm::for_each(json.meta.ipdf.callsites, [&](const auto& call_sites) {
    const auto& callee_function = retrieveFunction(data_map, call_sites.getKey());

    for (const auto& identifier : call_sites.getValue()) {
      data.callsite_callees.emplace(identifier.site_identifier, &callee_function);
    }
  });
}

static void insertArgumentFlow(const JsonACG::node_type& json, FunctionDescriptor& data, ACGDataMap& data_map) {
  llvm::for_each(json.meta.ipdf.inter_flow, [&](const auto& call_sites) {
    const auto& callee_function = retrieveFunction(data_map, call_sites.getKey());

    for (const auto& edge : call_sites.getValue()) {
      const FunctionDescriptor::ArgumentEdge flow_edge{edge.sink_argument_number, callee_function};
      data.reachable_function_arguments.emplace(edge.source_argument_number, flow_edge);
    }
  });
}

ACGDataMap createDatabase(const Regex& target_matcher, JsonACG& meta_cg) {
  ACGDataMap data_map;

  // insert nodes first, to avoid rehashing and invalidation of references
  llvm::for_each(meta_cg.function_nodes, [&](const auto& meta_cg_node) {
    data_map.try_emplace(meta_cg_node.getKey(), createFunctionNode(meta_cg_node.getValue(), target_matcher));
  });

  llvm::for_each(data_map, [&](auto& entry) {
    insertCallsiteIdentifiers(meta_cg.function_nodes[entry.getKey()], entry.getValue(), data_map);
    insertArgumentFlow(meta_cg.function_nodes[entry.getKey()], entry.getValue(), data_map);
  });

  return data_map;
}

/// tests if an edge can reach a pointer argument
inline static bool isSinkArgumentRelevant(const FunctionDescriptor::ArgumentEdge& edge) {
  return edge.callee.function_signature.paramIsType(edge.argument_number, [](const llvm::StringRef& type) {
    // this is a conservative implementation (currently structs with pointers are ignored)
    return type.endswith("*") || type == "ptr";
  });
}

/// an edge reaches a relevant sink argument
inline static bool edgeReachesRelevantSinkArgument(const FunctionDescriptor::ArgumentEdge& edge) {
  return edge.callee.is_target && isSinkArgumentRelevant(edge);
}

/// declaration with relevant sink argument, could reach indirect the destination
inline static bool edgeMaybeReachesSinkArgument(const FunctionDescriptor::ArgumentEdge& edge) {
  return !edge.callee.is_target && !edge.callee.is_definition && isSinkArgumentRelevant(edge);
}

static std::string calculateModuleHashSuffix(const llvm::Module& module) {
  llvm::SHA1 hasher;
  hasher.update(module.getSourceFileName());

  const std::string hex_hash = llvm::toHex(hasher.final());

  // Convert hash to Decimal to avoid problems with demangler
  // see also https://reviews.llvm.org/D93747
  const llvm::APInt int_hash{160, hex_hash, 16};

  // avoid using APInt::toString as it creates sometimes invalid results (unclear why this happens)
  llvm::SmallVector<char, 64> vec_hash;
  int_hash.toStringUnsigned(vec_hash, 10);

  const llvm::Twine hash_suffix = llvm::Twine("._argcg.") + vec_hash;
  return hash_suffix.str();
}

std::string FunctionIdentification::getIdentifierForFunction(const Function& function) const {
  if (!function.hasInternalLinkage()) {
    return function.getName().str();
  }

  return (llvm::Twine(function.getName()) + calculateModuleHashSuffix(*function.getParent())).str();
}

unsigned CallSiteIdentification::calculateCallsiteIdentifiersIfAbsent(const llvm::Function& function) {
  if (analyzedFunctions_.count(&function) != 0) {
    return analyzedFunctions_[&function];
  }

  unsigned call_site_identifier = 0;
  for (const auto& instruction : llvm::instructions(function)) {
    if (const auto& call_base = llvm::dyn_cast<llvm::CallBase>(&instruction)) {
      callSiteIdentifiers_[call_base] = ++call_site_identifier;
    }
  }

  analyzedFunctions_[&function] = call_site_identifier;
  return call_site_identifier;
}

unsigned CallSiteIdentification::getIdentifierForCallsite(const llvm::CallBase& site) const {
  const auto found_at = callSiteIdentifiers_.find(&site);
  assert((found_at != callSiteIdentifiers_.end()) && "identifier for callsite is missing");
  return found_at->second;
}

template <typename RangeT>
inline FilterAnalysis ACGFilterImpl::ACGFilterImpl::analyseMaybeCandidates(const RangeT&& maybe_candidates) const {
  bool has_continue = false;
  bool has_skip     = false;

  for (const auto& candidate : maybe_candidates) {
    switch (candidateMatcher_.match(candidate)) {
      case Matcher::MatchResult::Match:
        return FilterAnalysis::Keep;

      case Matcher::MatchResult::ShouldContinue:
        has_continue = true;
        break;

      case Matcher::MatchResult::ShouldSkip:
        has_skip = true;
        break;

      case Matcher::MatchResult::NoMatch:
        // keep the function if the matcher has no information
        return FilterAnalysis::Keep;
    }
  }

  // prioritize a continue
  if (has_continue) {
    return FilterAnalysis::Continue;
  }

  if (has_skip) {
    return FilterAnalysis::Skip;
  }

  // we have nothing found, use the default result
  return FilterAnalysis::Continue;
}

FilterAnalysis ACGFilterImpl::analyseFlowPath(
    const std::vector<FunctionDescriptor::ArgumentEdge>& initial_edges) const {
  enum class ReachabilityResult { kReaches, kMaybeReaches, kNeverReaches };
  using namespace typeart::filter::detail;

  /// contains declarations which have at least one argument that is reachable and relevant
  StringSet maybe_candidates;

  ReachabilityResult result = ReachabilityResult::kNeverReaches;
  solveReachable(initial_edges, [&](const FunctionDescriptor::ArgumentEdge& edge, const auto& enqueue) -> VisitResult {
    if (edgeReachesRelevantSinkArgument(edge)) {
      result = ReachabilityResult::kReaches;
      return kStop;
    }

    if (edgeMaybeReachesSinkArgument(edge)) {
      result = ReachabilityResult::kMaybeReaches;
      maybe_candidates.insert(edge.callee.function_signature.identifier);
    }

    // enqueue all edges that can be reached from the current callee/argument
    const auto& reachable_formals =
        make_range(edge.callee.reachable_function_arguments.equal_range(edge.argument_number));
    for (const auto& [ActualNumber, FormalEdge] : reachable_formals) {
      enqueue(FormalEdge);
    }

    return kContinue;
  });

  switch (result) {
    // check also the other call-sites
    case ReachabilityResult::kNeverReaches:
      return FilterAnalysis::Continue;

    case ReachabilityResult::kMaybeReaches:
      return analyseMaybeCandidates(maybe_candidates.keys());

    case ReachabilityResult::kReaches:
      return FilterAnalysis::Keep;
  }
}

/// calculates all outgoing edges for a given parameter value of a callsite
inline static std::vector<FunctionDescriptor::ArgumentEdge> createEdgesForCallsite(
    const llvm::CallBase& site, const llvm::Value& actual_argument,
    const std::vector<const FunctionDescriptor*>& callees_for_callsite) {
  const auto&& corresponding_args = llvm::make_filter_range(
      site.args(), [&actual_argument](const auto& arg) { return arg.get() == &actual_argument; });

  std::vector<FunctionDescriptor::ArgumentEdge> init;
  for (const auto& call_site_arg : corresponding_args) {
    for (const auto& callee : callees_for_callsite) {
      init.emplace_back(FunctionDescriptor::ArgumentEdge{
          static_cast<int>(call_site_arg.getOperandNo()) /*argumentNumber*/, *callee /*callee*/
      });
    }
  }

  return init;
}

inline static std::string prepareLogMessage(const llvm::CallBase& site, const llvm::Value& actual_argument,
                                            const llvm::StringRef& message_text) {
  std::string raw_string;
  raw_string_ostream string_stream{raw_string};
  string_stream << "function ";
  site.getFunction()->printAsOperand(string_stream, false);
  string_stream << ": argument ";
  actual_argument.printAsOperand(string_stream, false);
  string_stream << ": callsite ";
  site.print(string_stream, true);
  string_stream << " (";
  site.getDebugLoc().print(string_stream);
  string_stream << "): " << message_text;

  return raw_string;
}

inline static void logUnusedArgument(const llvm::CallBase& site, const llvm::Value& actual_argument) {
  if (site.getCalledOperand() == &actual_argument) {
    LOG_DEBUG(prepareLogMessage(site, actual_argument, "Argument is CalledOperand of Callsite"))
    return;
  }

  // this can happen when the IRSearch finds a store-instruction and adds
  // the pointer operand to the successors.
  if (llvm::is_contained(site.users(), &actual_argument)) {
    LOG_DEBUG(prepareLogMessage(site, actual_argument, "Argument is User of Callsite"))
    return;
  }

  // this is a serious problem within the dataflow analysis
  LOG_ERROR(prepareLogMessage(site, actual_argument, "Argument is not used at Callsite"))
}

inline static void logMissingCallees(const llvm::CallBase& site, const llvm::Value&) {
  std::string raw_string;
  raw_string_ostream string_stream{raw_string};
  string_stream << "function ";
  site.getFunction()->printAsOperand(string_stream, false);
  string_stream << ": no callees found for callsite ";
  site.print(string_stream, true);

  LOG_WARNING(raw_string)
}

inline static void logMissingEdges(const llvm::CallBase& site, const llvm::Value& actual_argument) {
  std::string raw_string;
  raw_string_ostream string_stream{raw_string};
  string_stream << "function ";
  site.getFunction()->printAsOperand(string_stream, false);
  string_stream << ": no edges found for argument ";
  actual_argument.printAsOperand(string_stream, false);
  string_stream << " @ ";
  site.print(string_stream, true);

  LOG_WARNING(raw_string)
}

FilterAnalysis ACGFilterImpl::analyseCallsite(const llvm::CallBase& site, const Path& path) const {
  const llvm::Value* actual_argument = path.getEndPrev().getValue();
  assert(actual_argument != nullptr && "Argument should not be null");

  if (!site.hasArgument(actual_argument)) {
    logUnusedArgument(site, *actual_argument);
    return FilterAnalysis::Continue;
  }

  /// the parent function of the callsite
  const auto& parent_function_name = functionIdentification_.getIdentifierForFunction(*site.getFunction());
  if (functionMap_.count(parent_function_name) == 0) {
    // This can happen when the (parent) function was not analyzed in the first place.
    // Which means that the parent function is not reachable from a program entry point. Either
    // the function is unreachable/dead or it was wrongly not analyzed (unsound). Therefore, we
    // conservative keep the value.
    LOG_INFO("function not in map: " << parent_function_name)
    return FilterAnalysis::Keep;
  }
  const auto& function_data = functionMap_.lookup(parent_function_name);

  const auto& callees = getCalleesForCallsite(function_data, site);
  if (callees.empty()) {
    // it is unlikely, but possible that a callsite was not analyzed in the first place.
    // this is often the result of an unsound analysis, therefore we want a conservative fallback
    logMissingCallees(site, *actual_argument);
    return FilterAnalysis::Keep;
  }

  const auto initial_edges = createEdgesForCallsite(site, *actual_argument, callees);
  if (initial_edges.empty()) {
    // no initial edges is likely an error in the callgraph file, keep the value in those cases
    logMissingEdges(site, *actual_argument);
    return FilterAnalysis::Keep;
  }

  return analyseFlowPath(initial_edges);
}

std::vector<const FunctionDescriptor*> ACGFilterImpl::getCalleesForCallsite(const FunctionDescriptor& function_data,
                                                                            const llvm::CallBase& site) const {
  const auto identifier       = callSiteIdentification_.getIdentifierForCallsite(site);
  const auto& map_entry_range = make_range(function_data.callsite_callees.equal_range(identifier));

  const auto& callees = map_range(map_entry_range, [](const auto& entry) { return entry.second; });

  return {callees.begin(), callees.end()};
}

FilterAnalysis ACGFilterImpl::precheck(llvm::Value* current_value, llvm::Function* start, const FPath& fpath) {
  assert(start != nullptr && "pre-check in FilterBase::DFSFuncFilter failed");

  // use the precheck-callback to identify all callsites within a function
  // preferably this would be implemented as a pass, but that would require
  // structural changes for TypeART.
  const auto number_of_call_sites = callSiteIdentification_.calculateCallsiteIdentifiersIfAbsent(*start);
  if (number_of_call_sites == 0) {
    return FilterAnalysis::Filter;
  }

  if (!fpath.empty()) {
    return FilterAnalysis::Continue;
  }

  // These conditions (temp alloc and alloca reaches task)
  // are only interesting if filter just started (aka fpath is empty)
  if (isTempAlloc(current_value)) {
    LOG_DEBUG("Alloca is a temporary " << *current_value);
    return FilterAnalysis::Filter;
  }

  if (auto* alloc = llvm::dyn_cast<AllocaInst>(current_value)) {
    if (alloc->getAllocatedType()->isStructTy() && omp::OmpContext::allocaReachesTask(alloc)) {
      LOG_DEBUG("Alloca reaches task call " << *alloc)
      return FilterAnalysis::Filter;
    }
  }

  return FilterAnalysis::Continue;
}

FilterAnalysis ACGFilterImpl::decl(const CallSite& site, const Path& path) {
  callSiteIdentification_.calculateCallsiteIdentifiersIfAbsent(*site->getFunction());
  return analyseCallsite(*llvm::cast<llvm::CallBase>(site.getInstruction()), path);
}

FilterAnalysis ACGFilterImpl::def(const CallSite& site, const Path& path) {
  callSiteIdentification_.calculateCallsiteIdentifiersIfAbsent(*site->getFunction());
  return analyseCallsite(*llvm::cast<llvm::CallBase>(site.getInstruction()), path);
}

FilterAnalysis ACGFilterImpl::indirect(const CallSite& site, const Path& path) {
  callSiteIdentification_.calculateCallsiteIdentifiersIfAbsent(*site->getFunction());
  return analyseCallsite(*llvm::cast<llvm::CallBase>(site.getInstruction()), path);
}

}  // namespace typeart::filter

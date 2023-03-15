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

using namespace llvm;

namespace typeart::filter {

namespace detail {

/// follows all enqueued values and visits them once starting with an initial entry

enum VisitResult {
  VR_Continue = 0,
  VR_Stop,
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
    const T* currentValue    = worklist.pop_back_val();
    const VisitResult status = callback(*currentValue, enqueue);
    switch (status) {
      case VR_Continue:
        break;
      case VR_Stop:
        return;
    }
  }
}
}  // namespace detail

static FunctionDescriptor createFunctionNode(const JSONACG::node_type& json, const llvm::Regex& targetMatcher) {
  const auto& signature          = json.meta.signature;
  const bool isTargetOfInterrest = targetMatcher.match(signature.identifier);

  return {isTargetOfInterrest,
          json.hasBody,
          {},
          {},
          {signature.identifier, signature.paramTypes, signature.returnType, signature.isVariadic}};
}

static const FunctionDescriptor& retrieveFunction(const ACGDataMap& dataMap, const llvm::StringRef& functionName) {
  assert(dataMap.count(functionName) && "Found an edge to an undefined callee");
  if (dataMap.count(functionName) == 0) {
    throw std::runtime_error("Found an edge to an undefined callee: " + functionName.str());
  }
  return dataMap.find(functionName)->second;
}

static void insertCallsiteIdentifiers(const JSONACG::node_type& json, FunctionDescriptor& data, ACGDataMap& dataMap) {
  llvm::for_each(json.meta.ipdf.callsites, [&](const auto& callSites) {
    const auto& calleeFunction = retrieveFunction(dataMap, callSites.getKey());

    for (const auto& identifier : callSites.getValue()) {
      data.callsiteCallees.emplace(identifier.siteIdentifier, &calleeFunction);
    }
  });
}

static void insertArgumentFlow(const JSONACG::node_type& json, FunctionDescriptor& data, ACGDataMap& dataMap) {
  llvm::for_each(json.meta.ipdf.interFlow, [&](const auto& callSites) {
    const auto& calleeFunction = retrieveFunction(dataMap, callSites.getKey());

    for (const auto& edge : callSites.getValue()) {
      FunctionDescriptor::ArgumentEdge const flowEdge{edge.sinkArgumentNumber, calleeFunction};
      data.reachableFunctionArguments.emplace(edge.sourceArgumentNumber, flowEdge);
    }
  });
}

ACGDataMap createDatabase(const Regex& targetMatcher, JSONACG& metaCg) {
  ACGDataMap dataMap;

  // insert nodes first, to avoid rehashing and invalidation of references
  llvm::for_each(metaCg.functionNodes, [&](const auto& metaCgNode) {
    dataMap.try_emplace(metaCgNode.getKey(), createFunctionNode(metaCgNode.getValue(), targetMatcher));
  });

  llvm::for_each(dataMap, [&](auto& entry) {
    insertCallsiteIdentifiers(metaCg.functionNodes[entry.getKey()], entry.getValue(), dataMap);
    insertArgumentFlow(metaCg.functionNodes[entry.getKey()], entry.getValue(), dataMap);
  });

  return dataMap;
}

/// tests if an edge can reach a pointer argument
inline static bool isSinkArgumentRelevant(const FunctionDescriptor::ArgumentEdge& edge) {
  return edge.callee.functionSignature.paramIsType(edge.argumentNumber, [](const llvm::StringRef& type) {
    // this is a conservative implementation (currently structs with pointers are ignored)
    return type.endswith("*") || type == "ptr";
  });
}

/// an edge reaches a relevant sink argument
inline static bool edgeReachesRelevantSinkArgument(const FunctionDescriptor::ArgumentEdge& edge) {
  return edge.callee.isTarget && isSinkArgumentRelevant(edge);
}

/// declaration with relevant sink argument, could reach indirect the destination
inline static bool edgeMaybeReachesSinkArgument(const FunctionDescriptor::ArgumentEdge& edge) {
  return !edge.callee.isTarget && !edge.callee.isDefinition && isSinkArgumentRelevant(edge);
}

template <typename RangeT>
inline FilterAnalysis ACGFilterImpl::ACGFilterImpl::analyseMaybeCandidates(const RangeT&& maybeCandidates) const {
  bool hasContinue = false;
  bool hasSkip     = false;

  for (const auto& candidate : maybeCandidates) {
    switch (candidateMatcher.match(candidate)) {
      case Matcher::MatchResult::Match:
        return FilterAnalysis::Keep;

      case Matcher::MatchResult::ShouldContinue:
        hasContinue = true;
        break;

      case Matcher::MatchResult::ShouldSkip:
        hasSkip = true;
        break;

      case Matcher::MatchResult::NoMatch:
        // keep the function if the matcher has no information
        return FilterAnalysis::Keep;
    }
  }

  // prioritize a continue
  if (hasContinue) {
    return FilterAnalysis::Continue;
  }

  if (hasSkip) {
    return FilterAnalysis::Skip;
  }

  // we have nothing found, use the default result
  return FilterAnalysis::Continue;
}

FilterAnalysis ACGFilterImpl::analyseFlowPath(const std::vector<FunctionDescriptor::ArgumentEdge>& initialEdges) const {
  enum class ReachabilityResult { reaches, maybe_reaches, never_reaches };
  using namespace typeart::filter::detail;

  /// contains declarations which have at least one argument that is reachable and relevant
  StringSet maybeCandidates;

  ReachabilityResult result = ReachabilityResult::never_reaches;
  solveReachable(initialEdges, [&](const FunctionDescriptor::ArgumentEdge& edge, const auto& enqueue) -> VisitResult {
    if (edgeReachesRelevantSinkArgument(edge)) {
      result = ReachabilityResult::reaches;
      return VR_Stop;
    }

    if (edgeMaybeReachesSinkArgument(edge)) {
      result = ReachabilityResult::maybe_reaches;
      maybeCandidates.insert(edge.callee.functionSignature.identifier);
    }

    // enqueue all edges that can be reached from the current callee/argument
    const auto& reachableFormals = make_range(edge.callee.reachableFunctionArguments.equal_range(edge.argumentNumber));
    for (const auto& [ActualNumber, FormalEdge] : reachableFormals) {
      enqueue(FormalEdge);
    }

    return VR_Continue;
  });

  switch (result) {
    // check also the other call-sites
    case ReachabilityResult::never_reaches:
      return FilterAnalysis::Continue;

    case ReachabilityResult::maybe_reaches:
      return analyseMaybeCandidates(maybeCandidates.keys());

    case ReachabilityResult::reaches:
      return FilterAnalysis::Keep;
  }
}

/// calculates all outgoing edges for a given parameter value of a callsite
inline static std::vector<FunctionDescriptor::ArgumentEdge> createEdgesForCallsite(
    const llvm::CallBase& site, const llvm::Value& actualArgument,
    const std::vector<const FunctionDescriptor*>& calleesForCallsite) {
  const auto&& correspondingArgs =
      llvm::make_filter_range(site.args(), [&actualArgument](const auto& arg) { return arg.get() == &actualArgument; });

  std::vector<FunctionDescriptor::ArgumentEdge> init;
  for (const auto& callSiteArg : correspondingArgs) {
    for (const auto& callee : calleesForCallsite) {
      init.emplace_back(FunctionDescriptor::ArgumentEdge{
          static_cast<int>(callSiteArg.getOperandNo()) /*argumentNumber*/, *callee /*callee*/
      });
    }
  }

  return init;
}

inline static std::string prepareLogMessage(const llvm::CallBase& site, const llvm::Value& actualArgument,
                                            const llvm::StringRef& msg) {
  std::string string;
  raw_string_ostream stringStream{string};
  stringStream << "function ";
  site.getFunction()->printAsOperand(stringStream, false);
  stringStream << ": argument ";
  actualArgument.printAsOperand(stringStream, false);
  stringStream << ": callsite ";
  site.print(stringStream, true);
  stringStream << " (";
  site.getDebugLoc().print(stringStream);
  stringStream << "): " << msg;

  return string;
}

inline static void logUnusedArgument(const llvm::CallBase& site, const llvm::Value& actualArgument) {
  if (site.getCalledOperand() == &actualArgument) {
    LOG_DEBUG(prepareLogMessage(site, actualArgument, "Argument is CalledOperand of Callsite"))
    return;
  }

  // this can happen when the IRSearch finds a store-instruction and adds
  // the pointer operand to the successors.
  if (llvm::is_contained(site.users(), &actualArgument)) {
    LOG_DEBUG(prepareLogMessage(site, actualArgument, "Argument is User of Callsite"))
    return;
  }

  // this is a serious problem within the dataflow analysis
  LOG_ERROR(prepareLogMessage(site, actualArgument, "Argument is not used at Callsite"))
}

inline static void logMissingCallees(const llvm::CallBase& site, const llvm::Value&) {
  std::string string;
  raw_string_ostream stringStream{string};
  stringStream << "function ";
  site.getFunction()->printAsOperand(stringStream, false);
  stringStream << ": no callees found for callsite ";
  site.print(stringStream, true);

  LOG_WARNING(string)
}

inline static void logMissingEdges(const llvm::CallBase& site, const llvm::Value& actualArgument) {
  std::string string;
  raw_string_ostream stringStream{string};
  stringStream << "function ";
  site.getFunction()->printAsOperand(stringStream, false);
  stringStream << ": no edges found for argument ";
  actualArgument.printAsOperand(stringStream, false);
  stringStream << " @ ";
  site.print(stringStream, true);

  LOG_WARNING(string)
}

FilterAnalysis ACGFilterImpl::analyseCallsite(const llvm::CallBase& site, const Path& path) const {
  const llvm::Value* actualArgument = path.getEndPrev().getValue();
  assert(actualArgument != nullptr && "Argument should not be null");

  if (!site.hasArgument(actualArgument)) {
    logUnusedArgument(site, *actualArgument);
    return FilterAnalysis::Continue;
  }

  /// the parent function of the callsite
  const auto& parentFunctionName = site.getFunction()->getName();
  if (functionMap.count(parentFunctionName) == 0) {
    // This can happen when the (parent) function was not analyzed in the first place.
    // Which means that the parent function is not reachable from a program entry point. Either
    // the function is unreachable/dead or it was wrongly not analyzed (unsound). Therefore, we
    // conservative keep the value.
    LOG_INFO("function not in map: " << parentFunctionName)
    return FilterAnalysis::Keep;
  }
  const auto& functionData = functionMap.lookup(parentFunctionName);

  const auto& callees = getCalleesForCallsite(functionData, site);
  if (callees.empty()) {
    // it is unlikely, but possible that a callsite was not analyzed in the first place.
    // this is often the result of an unsound analysis, therefore we want a conservative fallback
    logMissingCallees(site, *actualArgument);
    return FilterAnalysis::Keep;
  }

  const auto initialEdges = createEdgesForCallsite(site, *actualArgument, callees);
  if (initialEdges.empty()) {
    // no initial edges is likely an error in the callgraph file, keep the value in those cases
    logMissingEdges(site, *actualArgument);
    return FilterAnalysis::Keep;
  }

  return analyseFlowPath(initialEdges);
}

std::vector<const FunctionDescriptor*> ACGFilterImpl::getCalleesForCallsite(const FunctionDescriptor& functionData,
                                                                            const llvm::CallBase& site) const {
  const auto identifier     = getIdentifierForCallsite(site);
  const auto& mapEntryRange = make_range(functionData.callsiteCallees.equal_range(identifier));

  const auto& callees = map_range(mapEntryRange, [](const auto& entry) { return entry.second; });

  return {callees.begin(), callees.end()};
}

/// identifies all callsites of a function and returns the number of callsites of the given function
unsigned ACGFilterImpl::calculateSiteIdentifiersIfAbsent(const llvm::Function& function) {
  if (analyzedFunctions.count(&function) != 0) {
    return analyzedFunctions[&function];
  }

  unsigned callSiteIdentifier = 0;
  for (const auto& instruction : llvm::instructions(function)) {
    if (llvm::isa<llvm::CallBase>(&instruction)) {
      callSiteIdentifiers[&instruction] = ++callSiteIdentifier;
    }
  }

  analyzedFunctions[&function] = callSiteIdentifier;
  return callSiteIdentifier;
}

unsigned int ACGFilterImpl::getIdentifierForCallsite(const llvm::CallBase& site) const {
  const auto foundAt = callSiteIdentifiers.find(&site);
  assert((foundAt != callSiteIdentifiers.end()) && "identifier for callsite is missing");
  return foundAt->second;
}

FilterAnalysis ACGFilterImpl::precheck(llvm::Value* currentValue, llvm::Function* start, const FPath& fpath) {
  assert(start != nullptr && "pre-check in FilterBase::DFSFuncFilter failed");

  // use the precheck-callback to identify all callsites within a function
  // preferably this would be implemented as a pass, but that would require
  // structural changes for TypeART.
  const auto numberOfCallSites = calculateSiteIdentifiersIfAbsent(*start);
  if (numberOfCallSites == 0) {
    return FilterAnalysis::Filter;
  }

  if (!fpath.empty()) {
    return FilterAnalysis::Continue;
  }

  // These conditions (temp alloc and alloca reaches task)
  // are only interesting if filter just started (aka fpath is empty)
  if (isTempAlloc(currentValue)) {
    LOG_DEBUG("Alloca is a temporary " << *currentValue);
    return FilterAnalysis::Filter;
  }

  if (auto* alloc = llvm::dyn_cast<AllocaInst>(currentValue)) {
    if (alloc->getAllocatedType()->isStructTy() && omp::OmpContext::allocaReachesTask(alloc)) {
      LOG_DEBUG("Alloca reaches task call " << *alloc)
      return FilterAnalysis::Filter;
    }
  }

  return FilterAnalysis::Continue;
}

FilterAnalysis ACGFilterImpl::decl(const CallSite& site, const Path& path) {
  calculateSiteIdentifiersIfAbsent(*site->getFunction());
  return analyseCallsite(*llvm::cast<llvm::CallBase>(site.getInstruction()), path);
}

FilterAnalysis ACGFilterImpl::def(const CallSite& site, const Path& path) {
  calculateSiteIdentifiersIfAbsent(*site->getFunction());
  return analyseCallsite(*llvm::cast<llvm::CallBase>(site.getInstruction()), path);
}

FilterAnalysis ACGFilterImpl::indirect(const CallSite& site, const Path& path) {
  calculateSiteIdentifiersIfAbsent(*site->getFunction());
  return analyseCallsite(*llvm::cast<llvm::CallBase>(site.getInstruction()), path);
}

}  // namespace typeart::filter

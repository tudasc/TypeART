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
using namespace util::metacg;

namespace detail {

/// follows all enqueued values and visits them once starting with an initial entry

enum VisitResult {
  VR_Continue = 0,
  VR_Stop,
};

template <typename T, typename CB>
inline void solveReachable(const std::vector<T>& Range, CB&& Callback) noexcept {
  llvm::SmallPtrSet<T const*, 32> Visited{};
  llvm::SmallVector<T const*, 64> Worklist{};

  const auto& Enqueue = [&](T const& Entry) noexcept -> bool {
    if (Visited.insert(&Entry).second) {
      Worklist.push_back(&Entry);
      return true;
    }
    return false;
  };

  for (const auto& Item : Range) {
    Enqueue(Item);
  }

  while (!Worklist.empty()) {
    T const* CurrentValue    = Worklist.pop_back_val();
    VisitResult const Status = Callback(*CurrentValue, Enqueue);
    switch (Status) {
      case VR_Continue:
        break;
      case VR_Stop:
        return;
    }
  }
}
}  // namespace detail

static FunctionDescriptor createFunctionNode(const JSONACG::node_type& JSON, const Regex& TargetMatcher) {
  const auto& Signature          = JSON.meta.signature;
  const bool IsTargetOfInterrest = TargetMatcher.match(Signature.identifier);

  return {IsTargetOfInterrest,
          JSON.hasBody,
          {},
          {},
          {Signature.identifier, Signature.paramTypes, Signature.returnType, Signature.isVariadic}};
}

static FunctionDescriptor const& retrieveFunction(const ACGDataMap& DataMap, const StringRef& FunctionName) {
  assert(DataMap.count(FunctionName) && "Found an edge to an undefined callee");
  if (DataMap.count(FunctionName) == 0) {
    throw std::runtime_error("Found an edge to an undefined callee: " + FunctionName.str());
  }
  return DataMap.find(FunctionName)->second;
}

static void insertCallsiteIdentifiers(const JSONACG::node_type& Json, FunctionDescriptor& Data, ACGDataMap& DataMap) {
  llvm::for_each(Json.meta.ipdf.callsites, [&](const auto& CallSites) {
    const auto& CalleeFunction = retrieveFunction(DataMap, CallSites.getKey());

    for (const auto& Identifier : CallSites.getValue()) {
      Data.callsiteCallees.emplace(Identifier.id, &CalleeFunction);
    }
  });
}

static void insertArgumentFlow(const JSONACG::node_type& Json, FunctionDescriptor& Data, ACGDataMap& DataMap) {
  llvm::for_each(Json.meta.ipdf.interFlow, [&](const auto& CallSites) {
    const auto& CalleeFunction = retrieveFunction(DataMap, CallSites.getKey());

    for (const auto& Edge : CallSites.getValue()) {
      FunctionDescriptor::ArgumentEdge const FlowEdge{Edge.formalArgNo, CalleeFunction};
      Data.reachableFormals.emplace(Edge.actualArgNo, FlowEdge);
    }
  });
}

ACGDataMap createDatabase(const Regex& TargetMatcher, JSONACG& MetaCg) {
  ACGDataMap DataMap;

  // insert nodes first, to avoid rehashing and invalidation of references
  llvm::for_each(MetaCg.nodes, [&](const auto& MetaCGNode) {
    DataMap.try_emplace(MetaCGNode.getKey(), createFunctionNode(MetaCGNode.getValue(), TargetMatcher));
  });

  llvm::for_each(DataMap, [&](auto& Entry) {
    insertCallsiteIdentifiers(MetaCg.nodes[Entry.getKey()], Entry.getValue(), DataMap);
    insertArgumentFlow(MetaCg.nodes[Entry.getKey()], Entry.getValue(), DataMap);
  });

  return DataMap;
}

/// tests if an edge can reach a void* argument
bool ACGFilterImpl::isFormalArgumentRelevant(const FunctionDescriptor::ArgumentEdge& Edge) const {
  return Edge.callee.functionSignature.paramIsType(Edge.formalArgumentNumber, VoidType);
}

/// an edge reaches a relevant formal argument
bool ACGFilterImpl::edgeReachesRelevantFormalArgument(const FunctionDescriptor::ArgumentEdge& Edge) const {
  return Edge.callee.isTarget && isFormalArgumentRelevant(Edge);
}

/// declaration with relevant formal argument, could reach indirect the destination
bool ACGFilterImpl::edgeMaybeReachesFormalArgument(const FunctionDescriptor::ArgumentEdge& Edge) const {
  return !Edge.callee.isTarget && !Edge.callee.isDefinition && isFormalArgumentRelevant(Edge);
}

template <typename RangeT>
inline FilterAnalysis ACGFilterImpl::ACGFilterImpl::analyseMaybeCandidates(const RangeT&& MaybeCandidates) const {
  bool HasContinue = false;
  bool HasSkip     = false;

  for (const auto& Candidate : MaybeCandidates) {
    switch (candidateMatcher.match(Candidate)) {
      case Matcher::MatchResult::Match:
        return FilterAnalysis::Keep;

      case Matcher::MatchResult::ShouldContinue:
        HasContinue = true;
        break;

      case Matcher::MatchResult::ShouldSkip:
        HasSkip = true;
        break;

      case Matcher::MatchResult::NoMatch:
        // keep the function if the matcher has no information
        return FilterAnalysis::Keep;
    }
  }

  // prioritize a continue
  if (HasContinue) {
    return FilterAnalysis::Continue;
  }

  if (HasSkip) {
    return FilterAnalysis::Skip;
  }

  // we have nothing found, use the default result
  return FilterAnalysis::Continue;
}

FilterAnalysis ACGFilterImpl::analyseFlowPath(const std::vector<FunctionDescriptor::ArgumentEdge>& InitialEdges) const {
  enum class ReachabilityResult { reaches, maybe_reaches, never_reaches };
  using namespace typeart::filter::detail;

  /// contains declarations which have at least one argument that is reachable and relevant
  StringSet MaybeCandidates;

  ReachabilityResult Result = ReachabilityResult::never_reaches;
  solveReachable(InitialEdges, [&](const FunctionDescriptor::ArgumentEdge& Edge, auto const& Enqueue) -> VisitResult {
    if (edgeReachesRelevantFormalArgument(Edge)) {
      Result = ReachabilityResult::reaches;
      return VR_Stop;
    }

    if (edgeMaybeReachesFormalArgument(Edge)) {
      Result = ReachabilityResult::maybe_reaches;
      MaybeCandidates.insert(Edge.callee.functionSignature.identifier);
    }

    // enqueue all edges that can be reached from the current callee/argument
    const auto& ReachableFormals = make_range(Edge.callee.reachableFormals.equal_range(Edge.formalArgumentNumber));
    for (const auto& [ActualNumber, FormalEdge] : ReachableFormals) {
      Enqueue(FormalEdge);
    }

    return VR_Continue;
  });

  switch (Result) {
    // check also the other call-sites
    case ReachabilityResult::never_reaches:
      return FilterAnalysis::Continue;

    case ReachabilityResult::maybe_reaches:
      return analyseMaybeCandidates(MaybeCandidates.keys());

    case ReachabilityResult::reaches:
      return FilterAnalysis::Keep;
  }
}

/// calculates all outgoing edges for a given parameter value of a callsite
[[nodiscard]] std::vector<FunctionDescriptor::ArgumentEdge> ACGFilterImpl::createEdgesForCallsite(
    const CallBase& Site, const llvm::Value& ActualArgument,
    const std::vector<const FunctionDescriptor*>& CalleesForCallsite) const {
  const auto&& CorrespondingArgs =
      llvm::make_filter_range(Site.args(), [&ActualArgument](const auto& Arg) { return Arg.get() == &ActualArgument; });

  std::vector<FunctionDescriptor::ArgumentEdge> Init;
  for (const auto& CallSiteArg : CorrespondingArgs) {
    for (const auto& Callee : CalleesForCallsite) {
      Init.emplace_back(FunctionDescriptor::ArgumentEdge{
          static_cast<int>(CallSiteArg.getOperandNo()) /*formal*/, *Callee /*callee*/
      });
    }
  }

  return Init;
}

std::string ACGFilterImpl::prepareLogMessage(const CallBase& Site, const Value& ActualArgument,
                                             const StringRef& Msg) const {
  std::string String;
  raw_string_ostream StringStream{String};
  StringStream << "function ";
  Site.getFunction()->printAsOperand(StringStream, false);
  StringStream << ": argument ";
  ActualArgument.printAsOperand(StringStream, false);
  StringStream << ": callsite ";
  Site.print(StringStream, true);
  StringStream << " (";
  Site.getDebugLoc().print(StringStream);
  StringStream << "): " << Msg;

  return String;
}

inline void ACGFilterImpl::logUnusedArgument(const llvm::CallBase& Site, const llvm::Value& ActualArgument) const {
  if (Site.getCalledOperand() == &ActualArgument) {
    LOG_DEBUG(prepareLogMessage(Site, ActualArgument, "Argument is CalledOperand of Callsite"))
    return;
  }

  // this can happen when the IRSearch finds a store-instruction and adds
  // the pointer operand to the successors.
  if (llvm::is_contained(Site.users(), &ActualArgument)) {
    LOG_DEBUG(prepareLogMessage(Site, ActualArgument, "Argument is User of Callsite"))
    return;
  }

  // this is a serious problem within the dataflow analysis
  LOG_ERROR(prepareLogMessage(Site, ActualArgument, "Argument is not used at Callsite"))
}

inline void ACGFilterImpl::logMissingCallees(const llvm::CallBase& Site, const llvm::Value&) const {
  std::string String;
  raw_string_ostream StringStream{String};
  StringStream << "function ";
  Site.getFunction()->printAsOperand(StringStream, false);
  StringStream << ": no callees found for callsite ";
  Site.print(StringStream, true);

  LOG_WARNING(String)
}

inline void ACGFilterImpl::logMissingEdges(const llvm::CallBase& Site, const llvm::Value& ActualArgument) const {
  std::string String;
  raw_string_ostream StringStream{String};
  StringStream << "function ";
  Site.getFunction()->printAsOperand(StringStream, false);
  StringStream << ": no edges found for argument ";
  ActualArgument.printAsOperand(StringStream, false);
  StringStream << " @ ";
  Site.print(StringStream, true);

  LOG_WARNING(String)
}

FilterAnalysis ACGFilterImpl::analyseCallsite(const llvm::CallBase& Site, const Path& Path) const {
  const llvm::Value* ActualArgument = Path.getEndPrev().getValue();
  assert(ActualArgument != nullptr && "Argument should not be null");

  if (!Site.hasArgument(ActualArgument)) {
    logUnusedArgument(Site, *ActualArgument);
    return FilterAnalysis::Continue;
  }

  /// the parent function of the callsite
  const auto& ParentFunctionName = Site.getFunction()->getName();
  if (functionMap.count(ParentFunctionName) == 0) {
    // This can happen when the (parent) function was not analyzed in the first place.
    // Which means that the parent function is not reachable from a program entry point. Either
    // the function is unreachable/dead or it was wrongly not analyzed (unsound). Therefore, we
    // conservative keep the value.
    LOG_INFO("function not in map: " << ParentFunctionName)
    return typeart::filter::FilterAnalysis::Keep;
  }
  const auto& FunctionData = functionMap.lookup(ParentFunctionName);

  const auto& Callees = getCalleesForCallsite(FunctionData, Site);
  if (Callees.empty()) {
    // it is unlikely, but possible that a callsite was not analyzed in the first place.
    // this is often the result of an unsound analysis, therefore we want a conservative fallback
    logMissingCallees(Site, *ActualArgument);
    return FilterAnalysis::Keep;
  }

  const auto InitialEdges = createEdgesForCallsite(Site, *ActualArgument, Callees);
  if (InitialEdges.empty()) {
    // no initial edges is likely an error in the callgraph file, keep the value in those cases
    logMissingEdges(Site, *ActualArgument);
    return FilterAnalysis::Keep;
  }

  return analyseFlowPath(InitialEdges);
}

std::vector<const FunctionDescriptor*> ACGFilterImpl::getCalleesForCallsite(const FunctionDescriptor& FunctionData,
                                                                            const CallBase& Site) const {
  const auto Identifier     = getIdentifierForCallsite(Site);
  const auto& MapEntryRange = make_range(FunctionData.callsiteCallees.equal_range(Identifier));

  const auto& Callees = map_range(MapEntryRange, [](const auto& Entry) { return Entry.second; });

  return {Callees.begin(), Callees.end()};
}

/// identifiers all callsites of a function and stores additionally the highest used identifier as metadata field
/// at the function. if the function metadata field already exists, its value is returned
unsigned ACGFilterImpl::calculateSiteIdentifiersIfAbsent(const Function& Function) {
  if (analyzedFunctions.count(&Function) != 0) {
    return analyzedFunctions[&Function];
  }

  unsigned CallSiteIdentifier = 0;
  for (const auto& Instruction : llvm::instructions(Function)) {
    if (llvm::isa<llvm::CallBase>(&Instruction)) {
      callSiteIdentifiers[&Instruction] = ++CallSiteIdentifier;
    }
  }

  analyzedFunctions[&Function] = CallSiteIdentifier;
  return CallSiteIdentifier;
}

unsigned int ACGFilterImpl::getIdentifierForCallsite(const llvm::CallBase& Site) const {
  const auto FoundAt = callSiteIdentifiers.find(&Site);
  assert((FoundAt != callSiteIdentifiers.end()) && "identifier for callsite is missing");
  return FoundAt->second;
}

FilterAnalysis ACGFilterImpl::precheck(llvm::Value* In, llvm::Function* Start, const FPath& Fpath) {
  assert(Start != nullptr && "pre-check in FilterBase::DFSFuncFilter failed");

  // use the precheck-callback to identify all callsites within a function
  // preferably this would be implemented as a pass, but that would require
  // structural changes for TypeART.
  const auto NumberOfCallSites = calculateSiteIdentifiersIfAbsent(*Start);
  if (NumberOfCallSites == 0) {
    return FilterAnalysis::Filter;
  }

  if (!Fpath.empty()) {
    return FilterAnalysis::Continue;
  }

  // These conditions (temp alloc and alloca reaches task)
  // are only interesting if filter just started (aka fpath is empty)
  if (isTempAlloc(In)) {
    LOG_DEBUG("Alloca is a temporary " << *In);
    return FilterAnalysis::Filter;
  }

  if (auto* alloc = llvm::dyn_cast<AllocaInst>(In)) {
    if (alloc->getAllocatedType()->isStructTy() && omp::OmpContext::allocaReachesTask(alloc)) {
      LOG_DEBUG("Alloca reaches task call " << *alloc)
      return FilterAnalysis::Filter;
    }
  }

  return FilterAnalysis::Continue;
}

FilterAnalysis ACGFilterImpl::decl(const CallSite& Site, const Path& Path) {
  calculateSiteIdentifiersIfAbsent(*Site->getFunction());
  return analyseCallsite(*llvm::cast<llvm::CallBase>(Site.getInstruction()), Path);
}

FilterAnalysis ACGFilterImpl::def(const CallSite& Site, const Path& Path) {
  calculateSiteIdentifiersIfAbsent(*Site->getFunction());
  return analyseCallsite(*llvm::cast<llvm::CallBase>(Site.getInstruction()), Path);
}

FilterAnalysis ACGFilterImpl::indirect(const CallSite& Site, const Path& Path) {
  calculateSiteIdentifiersIfAbsent(*Site->getFunction());
  return analyseCallsite(*llvm::cast<llvm::CallBase>(Site.getInstruction()), Path);
}

}  // namespace typeart::filter
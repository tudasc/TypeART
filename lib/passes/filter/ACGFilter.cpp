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

#include "ACGFilter.h"

#include <llvm/ADT/SmallSet.h>

namespace typeart::filter {

AcgFilterImpl::AcgFilterImpl(metacg::Mcg&& cg, Regex&& match) : mcg{std::move(cg)}, matcher{std::move(match)} {
}

FilterAnalysis AcgFilterImpl::reachesMatching(const ArrayRef<size_t> nodes, const size_t idx) {
  SmallVector<std::pair<size_t, size_t>, 64> workq{};
  SmallSet<size_t, 32> seen{};

  const auto enqueue = [&seen, &workq](const size_t it, const size_t i) {
    if (const auto [_, inserted] = seen.insert(it); inserted) {
      workq.push_back({it, i});
    }
  };

  for (const auto id : nodes) {
    enqueue(id, idx);
    LOG_DEBUG("Starting node with parameter [" << idx << "]: " << mcg.forId(id)->name);
  }

  while (!workq.empty()) {
    // Grab the current node ID and translate it into its corresponding function descriptor to check
    // if its name matches our matcher.
    const auto [current, cur_idx] = workq.pop_back_val();
    LOG_DEBUG("> Inspecting node: " << mcg.forId(current)->name);

    if (const auto fn = mcg.forId(current); fn && fn->name) {
      if (matcher.match(*fn->name)) {
        // Keep if the function matches the matcher
        LOG_DEBUG("-> Matches matcher, keeping");
        return FilterAnalysis::Keep;
      } else if (const auto r = oracle.matchName(*fn->name); r != Matcher::MatchResult::NoMatch) {
        // Ignore any known skippable functions
        switch (r) {
          case Matcher::MatchResult::ShouldSkip:
          case Matcher::MatchResult::ShouldContinue:
            LOG_DEBUG("-> Known function, skipping");
            continue;

          default:;
        }
      }
    } else if (fn && !fn->has_body) {
      // We have to be conservative if we reach an unknown function without a body
      LOG_DEBUG("-> Function has no body, keeping");
      return FilterAnalysis::Keep;
    }

    const auto outs = mcg.outputs(current, cur_idx);
    if (!outs) {
      LOG_DEBUG("-> Failed to get outputs, possibly leaf");
      continue;
    }

    for (const auto& out : *outs) {
      for (const auto callee : out.callees) {
        enqueue(callee, out.idx);
        LOG_DEBUG("-> Enqueued callee: " << mcg.forId(callee)->name);
      }
    }
  }

  return FilterAnalysis::Continue;
}

FilterAnalysis AcgFilterImpl::precheck(Value* in, Function* start, const FPath&) {
  if (!start) {
    return FilterAnalysis::Continue;
  }

  FunctionAnalysis analysis{};
  analysis.analyze(start);

  // Filter if we're in a leaf function
  if (analysis.empty()) {
    return FilterAnalysis::Filter;
  }

  if (isTempAlloc(in)) {
    LOG_DEBUG("Alloca is a temporary " << *in);
    return FilterAnalysis::Filter;
  }

  if (AllocaInst* alloc = dyn_cast<AllocaInst>(in)) {
    if (alloc->getAllocatedType()->isStructTy() && omp::OmpContext::allocaReachesTask(alloc)) {
      LOG_DEBUG("Alloca reaches task call " << *alloc)
      return FilterAnalysis::Filter;
    }
  }

  return FilterAnalysis::Continue;
}

FilterAnalysis AcgFilterImpl::indirect(const CallSite current, const Path& p) {
  SmallVector<size_t, 16> callees{};

  const auto arg = *p.getEndPrev();
  assert(arg && "Argument is missing");

  if (!is_contained(current.args(), arg)) {
    return FilterAnalysis::Continue;
  }

  const auto idx = std::distance(current.args().begin(), find(current.args(), arg));

  const auto* callLoc = current.getLocation();
  if (!callLoc) {
    LOG_DEBUG("No call location, continuing");
    return FilterAnalysis::Continue;
  }

  // Resolve the parent scope via debug metadata as it should stay consistent even through inlining
  const auto* parentScope = dyn_cast<DISubprogram>(callLoc->getScope());
  if (!parentScope) {
    LOG_DEBUG("Failed to get parent scope, continuing");
    return FilterAnalysis::Continue;
  }

  const auto parentNode = mcg.byName(parentScope->getName());
  if (!parentNode) {
    LOG_DEBUG("Failed to get parent node, continuing");
    return FilterAnalysis::Continue;
  }

  const auto parent = mcg.forId(*parentNode);
  assert(parent && "Malformed MCG");

  auto md = parent->meta.as<metacg::MdLocals>("localflow");
  if (!md) {
    LOG_DEBUG("Failed to get localflow for node, continuing");
    return FilterAnalysis::Continue;
  }

  for (const auto& local : md->locals) {
    if (local.loc == metacg::SrcLoc{callLoc->getColumn(), callLoc->getLine()}) {
      callees.append(local.callees.begin(), local.callees.end());
    }
  }

  const auto outs = mcg.outputs(*parentNode, idx);
  if (!outs) {
    return FilterAnalysis::Continue;
  }

  for (const auto& out : *outs) {
    callees.append(out.callees.begin(), out.callees.end());
  }

  return reachesMatching(callees, idx);
}

FilterAnalysis AcgFilterImpl::def(const CallSite current, const Path& p) {
  const auto arg = *p.getEndPrev();
  assert(arg && "Argument is missing");

  if (!is_contained(current.args(), arg)) {
    return FilterAnalysis::Continue;
  }

  // Calculate the argument position
  const auto idx = std::distance(current.args().begin(), find(current.args(), arg));

  if (const auto node = mcg.byName(current.getCalledFunction()->getName()); node) {
    return reachesMatching({*node}, idx);
  } else {
    // Be conservative if the function is not recorded in the call graph
    LOG_DEBUG("Unrecorded function, continuing: " << current.getCalledFunction()->getName());
    return FilterAnalysis::Continue;
  }
}

FilterAnalysis AcgFilterImpl::decl(const CallSite current, const Path& p) {
  return def(current, p);
}

}  // namespace typeart::filter

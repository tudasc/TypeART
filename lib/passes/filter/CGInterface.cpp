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

#include "CGInterface.h"

#include "support/Logger.h"
#include "support/Util.h"

#include "llvm/ADT/Optional.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>

namespace typeart::filter {

CGInterface::ReachabilityResult JSONCG::reachable(const std::string& source, const std::string& target,
                                                  bool case_sensitive, bool short_circuit) {
  const auto reachables = get_reachable_functions(source);
  bool matches          = false;
  bool allBodies        = true;

  // bool what{false};
  if (!hasBodyMap[source]) {
    ++no_call_chain;
  }

  if (!reachables.empty()) {
    ++call_chain;
    f.push_back(source);
  }

  for (const auto& function_n : reachables) {
    matches |= util::regex_matches(target, function_n, case_sensitive);
    allBodies = allBodies && hasBodyMap[target];

    if (matches && short_circuit) {
      // we have found a match -> whether all functions have bodies is irrelevant
      return ReachabilityResult::reaches;
    }
  }

  if (matches) {
    // matches but no short circuit
    return ReachabilityResult::reaches;
  }

  if (!matches && (!allBodies || !hasBodyMap[source])) {
    // We did not find a match, but not all functions had bodies, we don't know
    return ReachabilityResult::maybe_reaches;
  }

  // No match and all functions had bodies -> never reaches (all call targets found)
  return ReachabilityResult::never_reaches;
}

std::vector<std::string> JSONCG::get_decl_only() {
  std::vector<std::string> list;
  list.reserve(hasBodyMap.size());
  for (const auto& [func, has_body] : hasBodyMap) {
    if (!has_body) {
      list.push_back(func);
    }
  }
  return list;
}

std::unordered_set<std::string> JSONCG::get_reachable_functions(const std::string& caller,
                                                                bool considerOverrides) const {
  std::unordered_set<std::string> ret;
  std::unordered_set<std::string> worklist;

  worklist = get_directly_called_function_names(caller, considerOverrides);
  while (!worklist.empty()) {
    const std::string func_name = *worklist.begin();
    // Check if we did not already handled it
    if (ret.find(func_name) == ret.end()) {
      worklist.merge(get_directly_called_function_names(func_name));
      ret.insert(func_name);
    }
    worklist.erase(worklist.find(func_name));  // Iterators get invalidated by merge, so we need to search again
  }
  return ret;
}

std::unordered_set<std::string> JSONCG::get_directly_called_function_names(const std::string& caller,
                                                                           bool considerOverrides) const {
  auto ref = directly_called_functions.find(caller);
  if (ref != std::end(directly_called_functions)) {
    // If the caller is virtual and overridden, add the overriding functions
    if (considerOverrides && (virtualTargets.find(caller) != std::end(virtualTargets))) {
      auto targets  = ref->second;
      auto vTargets = virtualTargets.find(caller)->second;
      targets.merge(vTargets);
      return targets;
    }
    return ref->second;
  }
  return {};
}

JSONCG::JSONCG(const llvm::json::Value& cg) {
  // Expected json format is the following:
  // A top level object/map Key is the function name, value is an object/map with information
  // We only care about "callees"
  // callees itself is an array with function names (as strings)
  assert(cg.kind() == llvm::json::Value::Kind::Object && "Top level json must be an Object");
  const llvm::json::Object* tlobj = cg.getAsObject();
  if (tlobj != nullptr) {
    for (const auto& entry : *tlobj) {
      // std::cout << "Building call site info for " << entry.first.str() << std::endl;
      construct_call_information(entry.first.str(), *tlobj);
    }
  }
}

void JSONCG::construct_call_information(const std::string& entry_caller, const llvm::json::Object& j) {
  if (directly_called_functions.find(entry_caller) == directly_called_functions.end()) {
    // We did not handle this function yet
    directly_called_functions[entry_caller] = std::unordered_set<std::string>();
    const auto caller                       = j.getObject(entry_caller);
    if (caller != nullptr) {
      const auto hasBody = caller->get("hasBody");
      if (hasBody != nullptr) {
        assert(hasBody->kind() == llvm::json::Value::Kind::Boolean && "hasBody must be boolean");
        hasBodyMap[entry_caller] = hasBody->getAsBoolean().getValue();
      }
      const auto calls = caller->getArray("callees");
      assert(calls != nullptr && "Json callee information is missing");
      if (calls != nullptr) {
        // Now iterate over them
        for (const auto& callee : *calls) {
          assert(callee.kind() == llvm::json::Value::Kind::String && "Callees must be strings");
          const auto callee_json_string = callee.getAsString();
          assert(callee_json_string.hasValue() && "Could not get callee as string");
          if (callee_json_string.hasValue()) {
            const std::string callee_string = std::string{callee_json_string.getValue()};
            directly_called_functions[entry_caller].insert(callee_string);
          }
        }
      }
      // if the function is virtual, overriding functions are _potential_ call targets
      const auto overridingFunctions = caller->getArray("overriddenBy");
      if (overridingFunctions != nullptr) {
        for (const auto& function : *overridingFunctions) {
          assert(function.kind() == llvm::json::Value::Kind::String && "Function names are always strings");
          const auto functionStr = function.getAsString();
          assert(functionStr.hasValue() && "Retrieving overriding function as String failed");
          if (functionStr.hasValue()) {
            const std::string functionName = std::string{functionStr.getValue()};
            virtualTargets[entry_caller].insert(functionName);
          }
        }
      }
    }
  }
}

std::unique_ptr<JSONCG> JSONCG::getJSON(const std::string& fileName) {
  using namespace llvm;
  auto memBuffer = MemoryBuffer::getFile(fileName);

  if (memBuffer) {
    auto json = llvm::json::parse(memBuffer.get()->getBuffer());

    if (!json) {
      std::string str;
      llvm::raw_string_ostream ostr(str);
      ostr << json.takeError();
      LOG_FATAL(ostr.str());
      exit(-1);
    }

    return std::make_unique<JSONCG>(json.get());
  }
  LOG_FATAL("No CG file provided / file cannot be found: " << fileName);
  return nullptr;
}

}  // namespace typeart::filter
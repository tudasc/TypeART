#include "CGInterface.h"

#include "Logger.h"
#include "Util.h"

#include <fstream>
#include <iostream>
#include <sstream>

namespace typeart {

bool JSONCG::reachable(const std::string& source, const std::string& target, bool case_sensitive, bool short_circuit) {
  const auto reachables = get_reachable_functions(source);
  bool matches          = false;
  for (const auto& f : reachables) {
    matches |= util::regex_matches(target, f, case_sensitive);
    if (matches && short_circuit) {
      return matches;
    }
  }
  return matches;
}

std::unordered_set<std::string> JSONCG::get_reachable_functions(const std::string& caller) const {
  std::unordered_set<std::string> ret;
  std::unordered_set<std::string> worklist;

  worklist = get_directly_called_function_names(caller);
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

std::unordered_set<std::string> JSONCG::get_directly_called_function_names(const std::string caller) const {
  auto ref = directly_called_functions.find(caller);
  if (ref != std::end(directly_called_functions)) {
    return ref->second;
  }
  return std::unordered_set<std::string>();
}

JSONCG::JSONCG(const llvm::json::Value& cg) {
  // Expected json format is the following:
  // A top level object/map Key is the function name, value is a object/map with informations
  // We only care about "callees"
  // callees itself ist an array with function names (as strings)
  assert(cg.kind() == llvm::json::Value::Kind::Object && "Top level json must be an Object");
  const llvm::json::Object* tlobj = cg.getAsObject();
  if (tlobj != nullptr) {
    for (const auto& entry : *tlobj) {
      std::cout << "Building call site info for " << entry.first.str() << std::endl;
      construct_call_information(entry.first.str(), *tlobj);
    }
  }
}

void JSONCG::construct_call_information(const std::string& entry, const llvm::json::Object& j) {
  if (directly_called_functions.find(entry) == directly_called_functions.end()) {
    // We did not handle this function yet
    directly_called_functions[entry] = std::unordered_set<std::string>();
    const auto caller                = j.getObject(entry);
    if (caller != nullptr) {
      const auto calles = caller->getArray("callees");
      assert(calles != nullptr && "Json callee information is missing");
      if (calles != nullptr) {
        // Now iterate over them
        for (const auto& callee : *calles) {
          assert(callee.kind() == llvm::json::Value::Kind::String && "Callees must be strings");
          const auto callee_json_string = callee.getAsString();
          assert(callee_json_string.hasValue() && "Could not get callee as string");
          if (callee_json_string.hasValue()) {
            const std::string callee_string = callee_json_string.getValue();
            directly_called_functions[entry].insert(callee_string);
          }
        }
      }
    }
  }
}

// llvm::json::Value& JSONCG::getJSON(const std::string& fileName) {
JSONCG* JSONCG::getJSON(const std::string& fileName) {
  std::ifstream inFile(fileName);
  if (inFile) {
    std::stringstream buf;
    buf << inFile.rdbuf();
    auto json = llvm::json::parse(buf.str());
    if (!json) {
      std::string str;
      llvm::raw_string_ostream ostr(str);
      ostr << json.takeError();
      LOG_FATAL(ostr.str());
      exit(-1);
    }

    return new JSONCG(json.get());
  } else {
    LOG_FATAL("No CG file provided / file cannot be found: " << fileName);
    exit(-1);
  }
}

}  // namespace typeart

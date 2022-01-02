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

#ifndef _LIB_TYPEART_CGINTERFACE_H
#define _LIB_TYPEART_CGINTERFACE_H

#include "llvm/Support/JSON.h"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace typeart::filter {

class CGInterface {
 public:
  enum class ReachabilityResult { reaches, maybe_reaches, never_reaches, unknown };

  CGInterface()                   = default;
  CGInterface(const CGInterface&) = default;
  CGInterface(CGInterface&&)      = default;
  CGInterface& operator=(const CGInterface&) = default;
  CGInterface& operator=(CGInterface&&) = default;

  /**
   * \brief Checks if a path exists from source to target
   */
  virtual ReachabilityResult reachable(const std::string& source, const std::string& target,
                                       bool case_sensitive = false, bool short_circuit = true) = 0;

  /**
   * \brief Returns all reachable functions starting from source
   */
  virtual std::unordered_set<std::string> get_reachable_functions(const std::string& source,
                                                                  bool considerOverrides) const = 0;

  virtual std::vector<std::string> get_decl_only() = 0;

  virtual ~CGInterface() = default;
};

class JSONCG final : public CGInterface {
  std::unordered_map<std::string, std::unordered_set<std::string>> directly_called_functions;
  std::unordered_map<std::string, bool> hasBodyMap;
  // in case a function is virtual, this map holds all potential overrides.
  std::unordered_map<std::string, std::unordered_set<std::string>> virtualTargets;
  size_t no_call_chain{0};
  size_t call_chain{0};
  std::vector<std::string> f;

 public:
  explicit JSONCG(const llvm::json::Value& cg);
  CGInterface::ReachabilityResult reachable(const std::string& source, const std::string& target,
                                            bool case_sensitive = false, bool short_circuit = true) override;

  std::unordered_set<std::string> get_reachable_functions(const std::string& source,
                                                          bool considerOverrides = true) const override;

  std::unordered_set<std::string> get_directly_called_function_names(const std::string& entry_caller,
                                                                     bool considerOverrides = true) const;

  std::vector<std::string> get_decl_only() override;

  static std::unique_ptr<JSONCG> getJSON(const std::string& fileName);

 private:
  void construct_call_information(const std::string& entry_caller, const llvm::json::Object& j);
};

}  // namespace typeart::filter
#endif
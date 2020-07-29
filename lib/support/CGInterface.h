#ifndef _LIB_TYPEART_CGINTERFACE_H
#define _LIB_TYPEART_CGINTERFACE_H

#include "llvm/Support/JSON.h"

#include <string>
#include <unordered_map>
#include <unordered_set>

namespace typeart {
class CGInterface {
 public:
  /**
   * \brief Checks if a path exists from source to target
   */
  virtual bool reachable(const std::string& source, const std::string& target, bool case_sensitive = false,
                         bool short_circuit = true) = 0;

  /**
   * \brief Returns all reachable functions starting from source
   */
  virtual std::unordered_set<std::string> get_reachable_functions(const std::string& source) const = 0;

  virtual ~CGInterface() = default;
};

class JSONCG : public CGInterface {
 public:
  explicit JSONCG(const llvm::json::Value& cg);
  bool reachable(const std::string& source, const std::string& target, bool case_sensitive = false,
                 bool short_circuit = true) override;
  std::unordered_set<std::string> get_reachable_functions(const std::string& source) const override;
  std::unordered_set<std::string> get_directly_called_function_names(const std::string caller) const;

  // static llvm::json::Value& getJSON(const std::string &fileName);
  static JSONCG* getJSON(const std::string& fileName);

  virtual ~JSONCG() = default;

 private:
  void construct_call_information(const std::string& caller, const llvm::json::Object& j);
  std::unordered_map<std::string, std::unordered_set<std::string>> directly_called_functions;
};
}  // namespace typeart
#endif

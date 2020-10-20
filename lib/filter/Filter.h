//
// Created by ahueck on 19.10.20.
//

#ifndef TYPEART_FILTER_H
#define TYPEART_FILTER_H

namespace llvm {
class Value;
class Function;
}  // namespace llvm

namespace typeart {
namespace filter {

class Filter {
 public:
  virtual bool filter(llvm::Value*)                 = 0;
  virtual void setStartingFunction(llvm::Function*) = 0;
  virtual void setMode(bool)                        = 0;
  virtual ~Filter()                                 = default;
};

class NoOpFilter final : public Filter {
 public:
  NoOpFilter() = default;
  bool filter(llvm::Value*) {
    return false;
  }
  void setMode(bool) {
  }
  void setStartingFunction(llvm::Function*) {
  }
  ~NoOpFilter() = default;
};

}  // namespace filter
}  // namespace typeart

#endif  // TYPEART_FILTER_H

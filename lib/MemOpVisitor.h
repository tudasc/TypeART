/*
 * MemOpVisitor.h
 *
 *  Created on: Jan 3, 2018
 *      Author: ahueck
 */

#ifndef LIB_MEMOPVISITOR_H_
#define LIB_MEMOPVISITOR_H_

#include "llvm/IR/InstVisitor.h"

#include <set>

namespace must {
namespace pass {

struct MemOpVisitor : public llvm::InstVisitor<MemOpVisitor> {
  MemOpVisitor();
  void visitCallInst(llvm::CallInst& ci);
  void visitMallocLike(llvm::CallInst& ci);
  void visitFreeLike(llvm::CallInst& ci);
  virtual ~MemOpVisitor();

 private:
  /** Look up sets for keyword strings */
  const std::set<std::string> allocFunctions{"malloc"};
  const std::set<std::string> deallocFunctions{"free"};
};

} /* namespace pass */
} /* namespace must */

#endif /* LIB_MEMOPVISITOR_H_ */

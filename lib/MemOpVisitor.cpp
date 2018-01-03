/*
 * MemOpVisitor.cpp
 *
 *  Created on: Jan 3, 2018
 *      Author: ahueck
 */

#include "MemOpVisitor.h"
#include "Logger.h"

namespace must {
namespace pass {

using namespace llvm;

MemOpVisitor::MemOpVisitor() = default;

void MemOpVisitor::visitCallInst(llvm::CallInst& ci) {
  const auto isInSet = [&ci](const auto& funcSet) {
    const auto name = ci.getCalledFunction()->getName().str();
    return std::find(std::begin(funcSet), std::end(funcSet), name) != std::end(funcSet);
  };

  if (isInSet(allocFunctions)) {
    visitMallocLike(ci);
  } else if (isInSet(deallocFunctions)) {
    visitFreeLike(ci);
  }
}

void MemOpVisitor::visitMallocLike(llvm::CallInst& ci) {
  LOG_DEBUG(ci.getCalledFunction()->getName());
}

void MemOpVisitor::visitFreeLike(llvm::CallInst& ci) {
  LOG_DEBUG(ci.getCalledFunction()->getName());
}

MemOpVisitor::~MemOpVisitor() = default;

} /* namespace pass */
} /* namespace must */

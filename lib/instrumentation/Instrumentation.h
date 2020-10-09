//
// Created by ahueck on 08.10.20.
//

#ifndef TYPEART_INSTRUMENTATION_H
#define TYPEART_INSTRUMENTATION_H

#include "../analysis/MemOpData.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"

namespace llvm {
class Value;
}

namespace typeart {

struct ArgMapper {
  using ArgsContainer = llvm::StringMap<llvm::Value*>;
  ArgsContainer args;

  ArgsContainer::mapped_type& operator[](ArgsContainer::key_type key) {
    return args[key];
  }

  /*
  const llvm::Optional<ArgsContainer::mapped_type> operator[](ArgsContainer::key_type key) const {
    auto it = args.find(key);
    if (it != args.end()) {
      return {it->second};
    }
    return llvm::None;
  }*/

  template <typename T>
  T* get_as(ArgsContainer::key_type key) const {
    T* elem{nullptr};
    if (auto it = args.find(key); it != args.end()) {
      elem = llvm::dyn_cast<T>(it->second);
    }
    return elem;
  }

  llvm::Value* get_value(ArgsContainer::key_type key) const {
    return get_as<llvm::Value>(key);
  }
};

namespace detail {
template <typename Data>
struct MemContainer {
  Data mem_data;
  ArgMapper args;
};
}  // namespace detail

using HeapContainer   = detail::MemContainer<MallocData>;
using FreeContainer   = detail::MemContainer<FreeData>;
using StackContainer  = detail::MemContainer<AllocaData>;
using GlobalContainer = detail::MemContainer<GlobalData>;

using HeapArgList   = llvm::SmallVector<HeapContainer, 16>;
using FreeArgList   = llvm::SmallVector<FreeContainer, 16>;
using StackArgList  = llvm::SmallVector<StackContainer, 16>;
using GlobalArgList = llvm::SmallVector<GlobalContainer, 8>;

class ArgumentCollector {
 public:
  virtual HeapArgList collectHeap(const llvm::SmallVectorImpl<MallocData>& mallocs)     = 0;
  virtual FreeArgList collectFree(const llvm::SmallVectorImpl<FreeData>& frees)         = 0;
  virtual StackArgList collectStack(const llvm::SmallVectorImpl<AllocaData>& frees)     = 0;
  virtual GlobalArgList collectGlobal(const llvm::SmallVectorImpl<GlobalData>& globals) = 0;
  virtual ~ArgumentCollector()                                                          = default;
};

class MemoryInstrument {
 public:
  virtual size_t instrumentHeap(const HeapArgList& heap)        = 0;
  virtual size_t instrumentFree(const FreeArgList& frees)       = 0;
  virtual size_t instrumentStack(const StackArgList& frees)     = 0;
  virtual size_t instrumentGlobal(const GlobalArgList& globals) = 0;
  virtual ~MemoryInstrument()                                   = default;
};

}  // namespace typeart

#endif  // TYPEART_INSTRUMENTATION_H

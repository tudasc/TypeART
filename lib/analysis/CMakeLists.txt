set(MEM_PASS_SOURCES
  MemInstFinderPass.cpp
  MemOpVisitor.cpp
  ../support/TypeUtil.cpp
)

make_llvm_module(MemInstFinderPass
  "${MEM_PASS_SOURCES}"
  INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
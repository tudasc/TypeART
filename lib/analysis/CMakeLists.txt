set(MEM_PASS_SOURCES
  MemInstFinderPass.cpp
  MemOpVisitor.cpp
)

make_llvm_module(meminstfinderpass
  "${MEM_PASS_SOURCES}"
  INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

install(
  TARGETS meminstfinderpass
  EXPORT typeart_llvm
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
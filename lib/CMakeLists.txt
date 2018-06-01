set(PASS_SOURCES
  MUSTSupportPass.cpp
  support/TypeUtil.cpp
  MemOpVisitor.cpp
  TypeManager.cpp
)

make_llvm_module(MustSupportPass
  "${PASS_SOURCES}"
  LINK_LIBS
    musttypelib
  INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/typelib
)

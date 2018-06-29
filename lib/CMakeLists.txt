add_subdirectory(analysis)

set(PASS_SOURCES
  TypeARTPass.cpp
  support/TypeUtil.cpp
  TypeManager.cpp
)

make_llvm_module(TypeArtPass
  "${PASS_SOURCES}"
  LINK_LIBS
    typelib
  DEPENDS
    MemInstFinderPass
  INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/typelib
)

install(
  TARGETS TypeArtPass
  EXPORT typeart_llvm
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

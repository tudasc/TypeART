set(MEM_PASS_SOURCES
  MemInstFinderPass.cpp
  MemOpVisitor.cpp
  ../support/TypeUtil.cpp
)

make_llvm_module(meminstfinderpass
  "${MEM_PASS_SOURCES}"
  LINK_LIBS
    memfilter
  INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/filter
)
mark_as_advanced(LLVM_MEMINSTFINDERPASS_LINK_INTO_TOOLS)

add_library(typeart::meminstfinderpass ALIAS meminstfinderpass)

target_compile_definitions(meminstfinderpass
  PRIVATE
  LOG_LEVEL=${LOG_LEVEL}
  $<$<BOOL:${SHOW_STATS}>:LLVM_ENABLE_STATS>
)

target_project_compile_options(meminstfinderpass)
target_project_coverage_options(meminstfinderpass)

install(
  TARGETS meminstfinderpass
  EXPORT ${TARGETS_EXPORT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
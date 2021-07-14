set(MEM_PASS_SOURCES
  MemInstFinderPass.cpp
  MemOpVisitor.cpp
  ../support/TypeUtil.cpp
)

make_llvm_module(LLVMMemOpFinderPass
  "${MEM_PASS_SOURCES}"
  LINK_LIBS
    typeart::LLVMMemOpFilter
  INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/filter
)
mark_as_advanced(LLVM_LLVMMEMOPFINDERPASS_LINK_INTO_TOOLS)

set_property(TARGET LLVMMemOpFinderPass PROPERTY OUTPUT_NAME "${PROJECT_NAME}LLVMMemOpFinderPass")
add_library(typeart::LLVMMemOpFinderPass ALIAS LLVMMemOpFinderPass)

target_compile_definitions(LLVMMemOpFinderPass
  PRIVATE
  LOG_LEVEL=${LOG_LEVEL}
  $<$<BOOL:${SHOW_STATS}>:LLVM_ENABLE_STATS>
)

target_project_compile_options(LLVMMemOpFinderPass)
target_project_coverage_options(LLVMMemOpFinderPass)

install(
  TARGETS LLVMMemOpFinderPass
  EXPORT ${TARGETS_EXPORT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
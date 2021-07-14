add_subdirectory(filter)
add_subdirectory(analysis)

set(PASS_SOURCES
  TypeARTPass.cpp
  support/TypeUtil.cpp
  instrumentation/InstrumentationHelper.cpp
  instrumentation/TypeARTFunctions.cpp
  TypeManager.cpp
  instrumentation/MemOpArgCollector.cpp
  instrumentation/MemOpInstrumentation.cpp
  instrumentation/Instrumentation.cpp
)

make_llvm_module(LLVMTransformPass
  "${PASS_SOURCES}"
  LINK_LIBS
    typeart::Types
  DEPENDS
    typeart::LLVMMemOpFinderPass
  INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/lib/
    ${PROJECT_SOURCE_DIR}/lib/passes/
)
mark_as_advanced(LLVM_LLVMTRANSFORMPASS_LINK_INTO_TOOLS)

set_property(TARGET LLVMTransformPass PROPERTY OUTPUT_NAME "${PROJECT_NAME}LLVMTransformPass")
add_library(typeart::LLVMTransformPass ALIAS LLVMTransformPass)

target_compile_definitions(LLVMTransformPass
  PRIVATE
  LOG_LEVEL=${LOG_LEVEL}
  $<$<BOOL:${SHOW_STATS}>:LLVM_ENABLE_STATS>
)

target_project_compile_options(LLVMTransformPass)
target_project_coverage_options(LLVMTransformPass)

install(
  TARGETS LLVMTransformPass
  EXPORT ${TARGETS_EXPORT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
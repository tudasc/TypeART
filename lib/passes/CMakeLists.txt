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

make_llvm_module(TransformPass
  "${PASS_SOURCES}"
  LINK_LIBS
    typeart::Types
  DEPENDS
    typeart::MemOpFinderPass
  INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/lib/
    ${PROJECT_SOURCE_DIR}/lib/passes/
)
mark_as_advanced(LLVM_TRANSFORMPASS_LINK_INTO_TOOLS)

set_property(TARGET TransformPass PROPERTY OUTPUT_NAME "${PROJECT_NAME}TransformPass")
add_library(typeart::TransformPass ALIAS TransformPass)

target_compile_definitions(TransformPass
  PRIVATE
  LOG_LEVEL=${LOG_LEVEL}
  $<$<BOOL:${SHOW_STATS}>:LLVM_ENABLE_STATS>
)

target_project_compile_options(TransformPass)
target_project_coverage_options(TransformPass)

install(
  TARGETS TransformPass
  EXPORT ${TARGETS_EXPORT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
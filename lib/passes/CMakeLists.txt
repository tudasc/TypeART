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

make_llvm_module(typeartpass
  "${PASS_SOURCES}"
  LINK_LIBS
    typelib
  DEPENDS
    meminstfinderpass
  INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/lib/
    ${PROJECT_SOURCE_DIR}/lib/passes/
)
mark_as_advanced(LLVM_TYPEARTPASS_LINK_INTO_TOOLS)

add_library(typeart::pass ALIAS typeartpass)

target_compile_definitions(typeartpass
  PRIVATE
  LOG_LEVEL=${LOG_LEVEL}
  $<$<BOOL:${SHOW_STATS}>:LLVM_ENABLE_STATS>
)

target_project_compile_options(typeartpass)
target_project_coverage_options(typeartpass)

install(
  TARGETS typeartpass
  EXPORT ${TARGETS_EXPORT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
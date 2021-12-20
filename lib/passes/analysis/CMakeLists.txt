set(MEM_PASS_SOURCES
  MemInstFinder.cpp
  MemOpVisitor.cpp
  ../support/TypeUtil.cpp
)

add_library(${TYPEART_PREFIX}_MemInstFinder STATIC ${MEM_PASS_SOURCES})
set_target_properties(
  ${TYPEART_PREFIX}_MemInstFinder
  PROPERTIES
  OUTPUT_NAME "${PROJECT_NAME}MemInstFinder"
  EXPORT_NAME "MemInstFinder"
)

add_library(typeart::MemInstFinder ALIAS ${TYPEART_PREFIX}_MemInstFinder)


target_compile_definitions(${TYPEART_PREFIX}_MemInstFinder PRIVATE LOG_LEVEL=${LOG_LEVEL})

set_target_properties(${TYPEART_PREFIX}_MemInstFinder PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(
  ${TYPEART_PREFIX}_MemInstFinder
  PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/filter>
)

target_include_directories(${TYPEART_PREFIX}_MemInstFinder SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})

make_tidy_check(${TYPEART_PREFIX}_MemInstFinder "${MEM_PASS_SOURCES}")

target_project_compile_options(${TYPEART_PREFIX}_MemInstFinder)
target_define_file_basename(${TYPEART_PREFIX}_MemInstFinder)
target_project_coverage_options(${TYPEART_PREFIX}_MemInstFinder)
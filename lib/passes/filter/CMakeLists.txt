set(FILTER_SOURCES
  CGInterface.cpp
  CGInterface.h
  Filter.h
  FilterBase.h
  CGForwardFilter.h
  CGForwardFilter.cpp
  StdForwardFilter.h
  StdForwardFilter.cpp
  FilterUtil.cpp
  FilterUtil.h
  IRSearch.h
)

add_library(${TYPEART_PREFIX}_MemOpFilter STATIC ${FILTER_SOURCES})
set_target_properties(
  ${TYPEART_PREFIX}_MemOpFilter
  PROPERTIES
  OUTPUT_NAME "${PROJECT_NAME}MemOpFilter"
  EXPORT_NAME "MemOpFilter"
)

add_library(typeart::MemOpFilter ALIAS ${TYPEART_PREFIX}_MemOpFilter)

# https://lists.llvm.org/pipermail/llvm-dev/2018-July/124534.html
# target_link_libraries(${TYPEART_PREFIX}_MemOpFilter PUBLIC LLVMCore LLVMSupport )

target_compile_definitions(${TYPEART_PREFIX}_MemOpFilter PRIVATE TYPEART_LOG_LEVEL=${TYPEART_LOG_LEVEL})

set_target_properties(${TYPEART_PREFIX}_MemOpFilter PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(
  ${TYPEART_PREFIX}_MemOpFilter ${warning_guard}
  PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib/passes>
         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
)

target_include_directories(${TYPEART_PREFIX}_MemOpFilter SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})

make_tidy_check(${TYPEART_PREFIX}_MemOpFilter "${FILTER_SOURCES}")

typeart_target_compile_options(${TYPEART_PREFIX}_MemOpFilter)
typeart_target_define_file_basename(${TYPEART_PREFIX}_MemOpFilter)
typeart_target_coverage_options(${TYPEART_PREFIX}_MemOpFilter)

# For now, non-consumable filter:
#install(
#  FILES Filter.h StandardFilter.h CGFilter.h
#  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
#)
#
#install(
#  TARGETS ${TYPEART_PREFIX}_MemOpFilter
#  EXPORT ${TARGETS_EXPORT_NAME}
#  LIBRARY DESTINATION lib
#  ARCHIVE DESTINATION lib
#)

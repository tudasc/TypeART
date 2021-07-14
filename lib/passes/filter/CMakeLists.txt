set(FILTER_SOURCES
        StandardFilter.cpp
        StandardFilter.h
        CGInterface.cpp
        CGInterface.h
        Filter.h
        FilterBase.h
        CGForwardFilter.h
        CGForwardFilter.cpp
        StdForwardFilter.h
        StdForwardFilter.cpp
        FilterUtil.cpp
        FilterUtil.h IRSearch.h)

add_library(LLVMMemOpFilter STATIC
  ${FILTER_SOURCES}
)
set_property(TARGET LLVMMemOpFilter PROPERTY OUTPUT_NAME "${PROJECT_NAME}LLVMMemOpFilter")
add_library(typeart::LLVMMemOpFilter ALIAS LLVMMemOpFilter)

# https://lists.llvm.org/pipermail/llvm-dev/2018-July/124534.html
#target_link_libraries(LLVMMemOpFilter PUBLIC
#  LLVMCore
#  LLVMSupport
#)

target_compile_definitions(LLVMMemOpFilter
  PRIVATE
  LOG_LEVEL=${LOG_LEVEL}
)

set_target_properties(LLVMMemOpFilter
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(LLVMMemOpFilter
  PUBLIC
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib/passes>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
)

target_include_directories(LLVMMemOpFilter
  SYSTEM
  PRIVATE
    ${LLVM_INCLUDE_DIRS}
)

make_tidy_check(LLVMMemOpFilter
  "${FILTER_SOURCES}"
)

target_project_compile_options(LLVMMemOpFilter)
target_define_file_basename(LLVMMemOpFilter)
target_project_coverage_options(LLVMMemOpFilter)

# For now, non-consumable filter
#install(
#  FILES Filter.h StandardFilter.h CGFilter.h
#  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
#)

#install(
#        TARGETS LLVMMemOpFilter
#        EXPORT ${TARGETS_EXPORT_NAME}
#        LIBRARY DESTINATION lib
#        ARCHIVE DESTINATION lib
#)
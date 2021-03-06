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

add_library(memfilter STATIC
        ${FILTER_SOURCES}
        )

# https://lists.llvm.org/pipermail/llvm-dev/2018-July/124534.html
#target_link_libraries(memfilter PUBLIC
#  LLVMCore
#  LLVMSupport
#)

target_project_compile_options(memfilter)

target_define_file_basename(memfilter)

target_project_compile_definitions(memfilter
        PRIVATE_DEFS
        LOG_LEVEL=${LOG_LEVEL}
        )

set_target_properties(memfilter
        PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        )


target_include_directories(memfilter
  PUBLIC
  ${PROJECT_SOURCE_DIR}/lib
  ${PROJECT_SOURCE_DIR}/lib/passes
  )

target_include_directories(memfilter
        SYSTEM
        PRIVATE
        ${LLVM_INCLUDE_DIRS}
        )

make_tidy_check(memfilter
        "${FILTER_SOURCES}"
        )

target_project_coverage_options(memfilter)

# For now, non-consumable filter
#install(
#  FILES Filter.h StandardFilter.h CGFilter.h
#  DESTINATION include
#)

#install(
#        TARGETS memfilter
#        EXPORT ${TARGETS_EXPORT_NAME}
#        LIBRARY DESTINATION lib
#        ARCHIVE DESTINATION lib
#)
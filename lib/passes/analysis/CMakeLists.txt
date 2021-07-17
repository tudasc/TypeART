set(MEM_PASS_SOURCES 
  MemInstFinderPass.cpp
  MemOpVisitor.cpp
  ../support/TypeUtil.cpp
)

make_llvm_module(
  MemOpFinderPass "${MEM_PASS_SOURCES}"
  LINK_LIBS typeart::MemOpFilter
  INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/lib
               ${CMAKE_CURRENT_SOURCE_DIR}/..
               ${CMAKE_CURRENT_SOURCE_DIR}/filter
)
mark_as_advanced(LLVM_MEMOPFINDERPASS_LINK_INTO_TOOLS)

set_property(
  TARGET MemOpFinderPass PROPERTY OUTPUT_NAME "${PROJECT_NAME}MemOpFinderPass"
)
add_library(typeart::MemOpFinderPass ALIAS MemOpFinderPass)

target_compile_definitions(
  MemOpFinderPass PRIVATE LOG_LEVEL=${LOG_LEVEL}
                          $<$<BOOL:${SHOW_STATS}>:LLVM_ENABLE_STATS>
)

target_project_compile_options(MemOpFinderPass)
target_project_coverage_options(MemOpFinderPass)

set(CONFIG_NAME ${PROJECT_NAME}MemOpFinderPass)
set(TARGETS_EXPORT_NAME ${CONFIG_NAME}Targets)

install(
  TARGETS MemOpFinderPass
  EXPORT ${TARGETS_EXPORT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
  EXPORT ${TARGETS_EXPORT_NAME}
  NAMESPACE typeart::
  DESTINATION ${TYPEART_INSTALL_CONFIGDIR}
)

export(
  EXPORT ${TARGETS_EXPORT_NAME}
  FILE ${CMAKE_BINARY_DIR}/${TARGETS_EXPORT_NAME}.cmake
  NAMESPACE typeart::
)

#configure_package_config_file(
#  ${PROJECT_SOURCE_DIR}/cmake/Config.cmake.in
#  ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}Config.cmake
#  INSTALL_DESTINATION ${TYPEART_INSTALL_CONFIGDIR}
#)
#
#install(FILES
#  ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}Config.cmake
#  DESTINATION ${TYPEART_INSTALL_CONFIGDIR}
#)

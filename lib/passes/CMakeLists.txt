add_subdirectory(typegen)
add_subdirectory(filter)
add_subdirectory(analysis)

set(PASS_SOURCES
  TypeARTPass.cpp
  support/TypeUtil.cpp
  instrumentation/InstrumentationHelper.cpp
  instrumentation/TypeARTFunctions.cpp
  instrumentation/MemOpArgCollector.cpp
  instrumentation/MemOpInstrumentation.cpp
  instrumentation/Instrumentation.cpp
)

typeart_make_llvm_module(
  ${TYPEART_PREFIX}_TransformPass "${PASS_SOURCES}"
  LINK_LIBS typeart::MemInstFinder
            typeart::TypesStatic
            typeart::TypeGenStatic
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
               ${PROJECT_SOURCE_DIR}/lib/
               ${PROJECT_SOURCE_DIR}/lib/passes/
               ${CMAKE_CURRENT_SOURCE_DIR}/typegen
)
mark_as_advanced(LLVM_TYPEART_TRANSFORMPASS_LINK_INTO_TOOLS)

set_target_properties(
  ${TYPEART_PREFIX}_TransformPass
  PROPERTIES
  OUTPUT_NAME "${PROJECT_NAME}TransformPass"
  EXPORT_NAME "TransformPass"
)
add_library(typeart::TransformPass ALIAS ${TYPEART_PREFIX}_TransformPass)

target_compile_definitions(
  ${TYPEART_PREFIX}_TransformPass PRIVATE TYPEART_LOG_LEVEL=${TYPEART_LOG_LEVEL}
                        $<$<BOOL:${TYPEART_SHOW_STATS}>:LLVM_FORCE_ENABLE_STATS=1>
)

typeart_target_compile_options(${TYPEART_PREFIX}_TransformPass)
typeart_target_coverage_options(${TYPEART_PREFIX}_TransformPass)

set(CONFIG_NAME ${PROJECT_NAME}TransformPass)
set(TARGETS_EXPORT_NAME ${CONFIG_NAME}Targets)

install(
  TARGETS ${TYPEART_PREFIX}_TransformPass
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

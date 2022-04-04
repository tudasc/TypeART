add_library(${TYPEART_PREFIX}_SystemStatic STATIC System.cpp)

add_library(typeart::SystemStatic ALIAS ${TYPEART_PREFIX}_SystemStatic)

set_target_properties(
  ${TYPEART_PREFIX}_SystemStatic
  PROPERTIES
  OUTPUT_NAME "${TYPEART_PREFIX}SystemStatic"
  EXPORT_NAME "SystemStatic"
  POSITION_INDEPENDENT_CODE ON
)

target_include_directories(
  ${TYPEART_PREFIX}_SystemStatic ${warning_guard}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>
)

typeart_target_compile_options(${TYPEART_PREFIX}_SystemStatic)
typeart_target_define_file_basename(${TYPEART_PREFIX}_SystemStatic)
typeart_target_coverage_options(${TYPEART_PREFIX}_SystemStatic)

if(TYPEART_TSAN)
  typeart_target_tsan_options(${TYPEART_PREFIX}_SystemStatic)
endif()

if(TYPEART_ASAN)
  typeart_target_asan_options(${TYPEART_PREFIX}_SystemStatic)
endif()

if(TYPEART_UBSAN)
  typeart_target_ubsan_options(${TYPEART_PREFIX}_SystemStatic)
endif()

set(CONFIG_NAME ${PROJECT_NAME}System)
set(TARGETS_EXPORT_NAME ${CONFIG_NAME}Targets)

install(FILES Table.h System.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/support
)

install(
  TARGETS ${TYPEART_PREFIX}_SystemStatic
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

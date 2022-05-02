set(TYPE_LIB_SOURCES TypeDB.cpp TypeIO.cpp)

add_library(${TYPEART_PREFIX}_TypesObj OBJECT ${TYPE_LIB_SOURCES})

set_target_properties(${TYPEART_PREFIX}_TypesObj PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(${TYPEART_PREFIX}_TypesObj SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})

target_include_directories(
  ${TYPEART_PREFIX}_TypesObj ${warning_guard}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
)

target_compile_definitions(${TYPEART_PREFIX}_TypesObj PRIVATE TYPEART_LOG_LEVEL=${TYPEART_LOG_LEVEL})

typeart_target_compile_options(${TYPEART_PREFIX}_TypesObj)

typeart_target_coverage_options(${TYPEART_PREFIX}_TypesObj)

make_tidy_check(${TYPEART_PREFIX}_TypesObj "${TYPE_LIB_SOURCES}")

add_library(${TYPEART_PREFIX}_Types SHARED $<TARGET_OBJECTS:${TYPEART_PREFIX}_TypesObj>)
add_library(typeart::Types ALIAS ${TYPEART_PREFIX}_Types)
set_target_properties(
  ${TYPEART_PREFIX}_Types
  PROPERTIES
  OUTPUT_NAME "${TYPEART_PREFIX}Types"
  EXPORT_NAME "Types"
)
target_include_directories(
  ${TYPEART_PREFIX}_Types ${warning_guard}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>
  PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
)

target_link_libraries(${TYPEART_PREFIX}_Types PRIVATE LLVMCore LLVMSupport)

typeart_target_coverage_options(${TYPEART_PREFIX}_Types)

add_library(${TYPEART_PREFIX}_TypesStatic STATIC $<TARGET_OBJECTS:${TYPEART_PREFIX}_TypesObj>)
add_library(typeart::TypesStatic ALIAS ${TYPEART_PREFIX}_TypesStatic)
set_target_properties(
  ${TYPEART_PREFIX}_TypesStatic
  PROPERTIES
  OUTPUT_NAME "${TYPEART_PREFIX}TypesStatic"
  EXPORT_NAME "TypesStatic"
)
target_include_directories(
  ${TYPEART_PREFIX}_TypesStatic ${warning_guard}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>
  PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
)

typeart_target_coverage_options(${TYPEART_PREFIX}_TypesStatic)

set(CONFIG_NAME ${PROJECT_NAME}Types)
set(TARGETS_EXPORT_NAME ${CONFIG_NAME}Targets)

install(FILES TypeInterface.h TypeDatabase.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)

install(
  TARGETS ${TYPEART_PREFIX}_Types ${TYPEART_PREFIX}_TypesStatic
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

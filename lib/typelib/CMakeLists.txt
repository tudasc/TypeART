set(TYPE_LIB_SOURCES TypeDB.cpp TypeIO.cpp)

add_library(TypesObj OBJECT ${TYPE_LIB_SOURCES})

set_target_properties(TypesObj PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(TypesObj SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})

target_include_directories(
  TypesObj
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
)

target_compile_definitions(TypesObj PRIVATE LOG_LEVEL=${LOG_LEVEL})

target_project_compile_options(TypesObj)
target_project_coverage_options(TypesObj)

make_tidy_check(TypesObj "${TYPE_LIB_SOURCES}")

add_library(Types SHARED $<TARGET_OBJECTS:TypesObj>)
add_library(typeart::Types ALIAS Types)
set_property(TARGET Types PROPERTY OUTPUT_NAME "${PROJECT_NAME}Types")
target_include_directories(
  Types
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
)

target_link_libraries(Types PRIVATE LLVMCore LLVMSupport)

add_library(TypesStatic STATIC $<TARGET_OBJECTS:TypesObj>)
add_library(typeart::TypesStatic ALIAS TypesStatic)
set_property(TARGET TypesStatic PROPERTY OUTPUT_NAME "${PROJECT_NAME}Types")
target_include_directories(
  TypesStatic
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
)

set(CONFIG_NAME ${PROJECT_NAME}Types)
set(TARGETS_EXPORT_NAME ${CONFIG_NAME}Targets)

install(FILES TypeInterface.h TypeDatabase.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)

install(
  TARGETS Types TypesStatic
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

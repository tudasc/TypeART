set(TYPEGEN_SOURCES
  TypeManager.cpp
  TypeManager.h
  TypeGenerator.h
  VectorTypeHandler.cpp
  VectorTypeHandler.h
  StructTypeHandler.cpp
  StructTypeHandler.h
  ../support/TypeUtil.cpp
)

add_library(${TYPEART_PREFIX}_TypeGenObj OBJECT ${TYPEGEN_SOURCES})

target_compile_definitions(${TYPEART_PREFIX}_TypeGenObj PRIVATE TYPEART_LOG_LEVEL=${TYPEART_LOG_LEVEL})

set_target_properties(${TYPEART_PREFIX}_TypeGenObj PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(
  ${TYPEART_PREFIX}_TypeGenObj ${warning_guard}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib/passes>
         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib/typelib>
)

target_include_directories(${TYPEART_PREFIX}_TypeGenObj SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})

make_tidy_check(${TYPEART_PREFIX}_TypeGenObj "${TypeGen_SOURCES}")

typeart_target_compile_options(${TYPEART_PREFIX}_TypeGenObj)
typeart_target_define_file_basename(${TYPEART_PREFIX}_TypeGenObj)
typeart_target_coverage_options(${TYPEART_PREFIX}_TypeGenObj)

# Need to link LLVMCore to work for consumer
add_library(${TYPEART_PREFIX}_TypeGenStatic STATIC $<TARGET_OBJECTS:${TYPEART_PREFIX}_TypeGenObj>)
add_library(typeart::TypeGenStatic ALIAS ${TYPEART_PREFIX}_TypeGenStatic)

set_target_properties(
  ${TYPEART_PREFIX}_TypeGenStatic
  PROPERTIES
  OUTPUT_NAME "${TYPEART_PREFIX}TypeGen"
  EXPORT_NAME "TypeGenStatic"
)

typeart_target_coverage_options(${TYPEART_PREFIX}_TypeGenStatic)

target_link_libraries(${TYPEART_PREFIX}_TypeGenStatic typeart::TypesStatic)

target_include_directories(
  ${TYPEART_PREFIX}_TypeGenStatic ${warning_guard}
  PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib/passes>
)

set(CONFIG_NAME ${TYPEART_PREFIX}TypeGen)
set(TARGETS_EXPORT_NAME ${CONFIG_NAME}Targets)

install(FILES TypeGenerator.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
  )

install(
  TARGETS ${TYPEART_PREFIX}_TypeGenStatic
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
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

add_library(TypeGenObj OBJECT ${TYPEGEN_SOURCES})

target_compile_definitions(TypeGenObj PRIVATE LOG_LEVEL=${LOG_LEVEL})

set_target_properties(TypeGenObj PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(
  TypeGenObj
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib/passes>
)

target_include_directories(TypeGenObj SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})

make_tidy_check(TypeGenObj "${TypeGen_SOURCES}")

target_project_compile_options(TypeGenObj)
target_define_file_basename(TypeGenObj)
target_project_coverage_options(TypeGenObj)

# Need to link LLVMCore to work for consumer
add_library(TypeGenStatic STATIC $<TARGET_OBJECTS:TypeGenObj>)
add_library(typeart::TypeGenStatic ALIAS TypeGenStatic)
set_property(
  TARGET TypeGenStatic PROPERTY OUTPUT_NAME "${PROJECT_NAME}TypeGen"
)
target_include_directories(
  TypeGenStatic
  PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib/passes>
)

set(CONFIG_NAME ${PROJECT_NAME}TypeGen)
set(TARGETS_EXPORT_NAME ${CONFIG_NAME}Targets)

install(FILES TypeGenerator.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
  )

install(
  TARGETS TypeGenStatic
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
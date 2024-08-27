set(TYPEGEN_DIMETA_SOURCES
  ../TypeGenerator.h
  ../TypeIDGenerator.h
  ../TypeGenerator.cpp
  DimetaTypeGen.cpp
)

add_library(${TYPEART_PREFIX}_TypeGenDimetaStatic OBJECT ${TYPEGEN_DIMETA_SOURCES})

target_compile_definitions(${TYPEART_PREFIX}_TypeGenDimetaStatic PRIVATE TYPEART_LOG_LEVEL=${TYPEART_LOG_LEVEL})

set_target_properties(${TYPEART_PREFIX}_TypeGenDimetaStatic PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(
  ${TYPEART_PREFIX}_TypeGenDimetaStatic ${warning_guard}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib/passes>
)

target_include_directories(${TYPEART_PREFIX}_TypeGenDimetaStatic SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})

make_tidy_check(${TYPEART_PREFIX}_TypeGenDimetaStatic "${TYPEGEN_DIMETA_SOURCES}")

typeart_target_compile_options(${TYPEART_PREFIX}_TypeGenDimetaStatic)
typeart_target_define_file_basename(${TYPEART_PREFIX}_TypeGenDimetaStatic)
typeart_target_coverage_options(${TYPEART_PREFIX}_TypeGenDimetaStatic)

# Need to link LLVMCore to work for consumer
# add_library(${TYPEART_PREFIX}_TypeGenDimetaStatic STATIC $<TARGET_OBJECTS:${TYPEART_PREFIX}_TypeGenDimetaStatic>)
add_library(typeart::TypeGenDimetaStatic ALIAS ${TYPEART_PREFIX}_TypeGenDimetaStatic)

set_target_properties(
  ${TYPEART_PREFIX}_TypeGenDimetaStatic
  PROPERTIES
  OUTPUT_NAME "${TYPEART_PREFIX}TypeGenDimeta"
  EXPORT_NAME "TypeGenDimetaStatic"
)

typeart_target_coverage_options(${TYPEART_PREFIX}_TypeGenDimetaStatic)

target_link_libraries(${TYPEART_PREFIX}_TypeGenDimetaStatic PUBLIC typeart::TypesStatic)
target_link_libraries(${TYPEART_PREFIX}_TypeGenDimetaStatic PUBLIC dimeta::Types)

target_include_directories(
  ${TYPEART_PREFIX}_TypeGenDimetaStatic ${warning_guard}
  PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib/passes>
)

# set(CONFIG_NAME ${TYPEART_PREFIX}TypeGenDimeta)
# set(TARGETS_EXPORT_NAME ${CONFIG_NAME}Targets)

# install(FILES TypeGenerator.h
#   DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
#   )

# install(
#   TARGETS ${TYPEART_PREFIX}_TypeGenDimetaStatic
#   EXPORT ${TARGETS_EXPORT_NAME}
#   LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#   ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
# )

# install(
#   EXPORT ${TARGETS_EXPORT_NAME}
#   NAMESPACE typeart::
#   DESTINATION ${TYPEART_INSTALL_CONFIGDIR}
# )

# export(
#   EXPORT ${TARGETS_EXPORT_NAME}
#   FILE ${CMAKE_BINARY_DIR}/${TARGETS_EXPORT_NAME}.cmake
#   NAMESPACE typeart::
# )

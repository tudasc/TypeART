set(CONFIG_SOURCES
  FileConfiguration.cpp
  EnvironmentConfiguration.cpp
  TypeARTOptions.cpp
)

add_library(${TYPEART_PREFIX}_PassConfiguration STATIC ${CONFIG_SOURCES})

set_target_properties(
  ${TYPEART_PREFIX}_PassConfiguration
  PROPERTIES
  OUTPUT_NAME "${PROJECT_NAME}PassConfiguration"
  EXPORT_NAME "PassConfiguration"
)

add_library(typeart::PassConfiguration ALIAS ${TYPEART_PREFIX}_PassConfiguration)

target_compile_definitions(
  ${TYPEART_PREFIX}_PassConfiguration PRIVATE TYPEART_LOG_LEVEL=${TYPEART_LOG_LEVEL}
                                              LLVM_VERSION_MAJOR=${LLVM_VERSION_MAJOR}
)

set_target_properties(${TYPEART_PREFIX}_PassConfiguration PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(
  ${TYPEART_PREFIX}_PassConfiguration ${warning_guard}
  PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>
         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib/passes>
)

target_include_directories(${TYPEART_PREFIX}_PassConfiguration SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})

make_tidy_check(${TYPEART_PREFIX}_PassConfiguration "${CONFIG_SOURCES}")

typeart_target_compile_options(${TYPEART_PREFIX}_PassConfiguration)
typeart_target_define_file_basename(${TYPEART_PREFIX}_PassConfiguration)
typeart_target_coverage_options(${TYPEART_PREFIX}_PassConfiguration)

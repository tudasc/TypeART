set(TYPE_LIB_SOURCES
  TypeDB.cpp
  TypeIO.cpp
)

add_library(Types STATIC
  ${TYPE_LIB_SOURCES}
)
set_property(TARGET Types PROPERTY OUTPUT_NAME "${PROJECT_NAME}Types")
add_library(typeart::Types ALIAS Types)

set_target_properties(Types
  PROPERTIES POSITION_INDEPENDENT_CODE ON
)

target_include_directories(Types
  SYSTEM 
  PRIVATE 
    ${LLVM_INCLUDE_DIRS}
)

target_include_directories(Types
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
)

target_compile_definitions(Types
  PRIVATE
    LOG_LEVEL=${LOG_LEVEL}
)

target_project_compile_options(Types)
target_project_coverage_options(Types)

make_tidy_check(Types
  "${TYPE_LIB_SOURCES}"
)

install(
  TARGETS Types
  EXPORT ${TARGETS_EXPORT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
  FILES TypeInterface.h TypeIO.h TypeDB.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)

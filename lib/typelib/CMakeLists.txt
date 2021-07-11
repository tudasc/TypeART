set(PROJECT_NAME typeart_typelib)
set(TARGETS_EXPORT_NAME ${PROJECT_NAME}-targets)

set(TYPE_LIB_SOURCES
  TypeDB.cpp 
  TypeIO.cpp
)

add_library(typelib STATIC
  ${TYPE_LIB_SOURCES}
)

add_library(typeart::typelib ALIAS typelib)

#target_link_libraries(musttypelib
#  LLVMCore
#  LLVMSupport
#)

set_target_properties(typelib
  PROPERTIES POSITION_INDEPENDENT_CODE ON
)

target_include_directories(typelib
  SYSTEM 
  PRIVATE 
    ${LLVM_INCLUDE_DIRS}
)

target_include_directories(typelib
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
  PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
)

target_compile_definitions(memfilter
  PRIVATE
    LOG_LEVEL=${LOG_LEVEL}
)

target_project_compile_options(typelib)
target_project_coverage_options(typelib)

make_tidy_check(typelib
  "${TYPE_LIB_SOURCES}"
)

install(
  TARGETS typelib
  EXPORT ${TARGETS_EXPORT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

install(
  FILES TypeInterface.h TypeIO.h TypeDB.h
  DESTINATION include
)

install(
  EXPORT ${TARGETS_EXPORT_NAME}
  NAMESPACE typeart::
  DESTINATION lib/cmake
)

configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION lib/cmake
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  DESTINATION lib/cmake
)
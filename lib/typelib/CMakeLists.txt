set(PROJECT_NAME typeart_typelib)
set(TARGETS_EXPORT_NAME ${PROJECT_NAME}-targets)

set(TYPE_LIB_SOURCES
  TypeDB.cpp 
  TypeIO.cpp
)

add_library(typelib STATIC
  ${TYPE_LIB_SOURCES}
)

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
  PRIVATE
  ${PROJECT_SOURCE_DIR}/lib
)

target_project_coverage_options(typelib)

target_project_compile_options(typelib)
target_project_compile_definitions(typelib
  ARG_PRIVATE_DEFS
    LOG_LEVEL=${LOG_LEVEL}
)

make_tidy_check(typelib
  "${TYPE_LIB_SOURCES}"
)

install(
  TARGETS typelib
  EXPORT ${TARGETS_EXPORT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  INCLUDES DESTINATION
    include 
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
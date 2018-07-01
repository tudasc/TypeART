set(RUNTIME_LIB_SOURCES
  Runtime.cpp
)

add_library(runtime SHARED
  ${RUNTIME_LIB_SOURCES}
)

target_link_libraries(runtime 
  typelib
)

target_include_directories(runtime 
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/typelib>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_SOURCE_DIR}/lib # For the logger
)

target_include_directories(runtime 
  SYSTEM 
  PRIVATE
    ${LLVM_INCLUDE_DIRS}
)

target_define_file_basename(runtime)

target_link_libraries(runtime
  LLVMCore
  LLVMSupport
)

make_tidy_check(runtime
  "${RUNTIME_LIB_SOURCES}"
)


set(PROJECT_NAME typeart_runtime)
set(TARGETS_EXPORT_NAME typeart_runtime)

install(
  TARGETS runtime
  EXPORT ${TARGETS_EXPORT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
#  INCLUDES DESTINATION
#    include # --> superfluous since $<INSTALL_INTERFACE:include> is used
)

install(
  FILES RuntimeInterface.h
  DESTINATION include
)

install(
  EXPORT ${TARGETS_EXPORT_NAME}
  NAMESPACE typeart::
  DESTINATION lib/cmake
)

#set(DEPENDENCY libaTargets.cmake
configure_package_config_file(
  ${CMAKE_SOURCE_DIR}/cmake/Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/typeart_runtimeConfig.cmake
  INSTALL_DESTINATION lib/cmake
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/typeart_runtimeConfig.cmake
  DESTINATION lib/cmake
)
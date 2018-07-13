set(PROJECT_NAME typeart_runtime)
set(TARGETS_EXPORT_NAME ${PROJECT_NAME}-targets)

set(RUNTIME_LIB_SOURCES
  Runtime.cpp
)

if(MPI_LOGGER)
  list(APPEND RUNTIME_LIB_SOURCES MPILogger.cpp)
endif()

add_library(runtime SHARED
  ${RUNTIME_LIB_SOURCES}
)

set(RUNTIME_LIB_DEPS typelib)
if(MPI_LOGGER)
  list(APPEND RUNTIME_LIB_DEPS ${MPI_LIBRARIES})
  target_project_compile_definitions(runtime
    PRIVATE_DEFS MPI_LOGGER=1
  )
endif()

target_link_libraries(runtime 
  ${RUNTIME_LIB_DEPS}
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
    ${MPI_C_INCLUDE_PATH}
)

target_project_compile_options(runtime)
target_project_compile_definitions(runtime)
target_define_file_basename(runtime)


target_link_libraries(runtime
  LLVMCore
  LLVMSupport
)

make_tidy_check(runtime
  "${RUNTIME_LIB_SOURCES}"
)


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
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION lib/cmake
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  DESTINATION lib/cmake
)

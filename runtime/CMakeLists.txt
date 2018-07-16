set(PROJECT_NAME typeart-rt)
set(TARGETS_EXPORT_NAME ${PROJECT_NAME}-targets)

set(RUNTIME_LIB_SOURCES
  Runtime.cpp
)

if(MPI_LOGGER)
  list(APPEND RUNTIME_LIB_SOURCES MPILogger.cpp)
endif()

add_library(typeart-rt SHARED
  ${RUNTIME_LIB_SOURCES}
)

set(RUNTIME_LIB_DEPS typelib)
if(MPI_LOGGER)
  list(APPEND RUNTIME_LIB_DEPS ${MPI_LIBRARIES})
  target_project_compile_definitions(typeart-rt
    PRIVATE_DEFS MPI_LOGGER=1
  )
endif()

target_link_libraries(typeart-rt 
  ${RUNTIME_LIB_DEPS}
)

target_include_directories(typeart-rt 
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/runtime>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/typelib>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${PROJECT_SOURCE_DIR}/lib # For the logger
)

target_include_directories(typeart-rt
  SYSTEM 
  PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${MPI_C_INCLUDE_PATH}
)

target_project_compile_options(typeart-rt)
target_project_compile_definitions(typeart-rt)
target_define_file_basename(typeart-rt)

target_link_libraries(typeart-rt
  LLVMCore
  LLVMSupport
)

make_tidy_check(typeart-rt
  "${RUNTIME_LIB_SOURCES}"
)


install(
  TARGETS typeart-rt
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
  ${PROJECT_SOURCE_DIR}/cmake/Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION lib/cmake
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  DESTINATION lib/cmake
)

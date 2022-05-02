set(TYPEART_META_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/Version.cpp)

add_custom_command(
  OUTPUT ${TYPEART_META_SOURCE}
  COMMAND ${CMAKE_COMMAND}
          -DROOT_DIR=${PROJECT_SOURCE_DIR}
          -DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}
          -DPROJECT_VERSION=${PROJECT_VERSION}
          -DLLVM_VERSION_MAJOR=${LLVM_VERSION_MAJOR}
          -DLLVM_VERSION_MINOR=${LLVM_VERSION_MINOR}
          -P ${PROJECT_SOURCE_DIR}/cmake/modules/version-helper.cmake
  MAIN_DEPENDENCY Version.cpp.in
  DEPENDS ${PROJECT_SOURCE_DIR}/cmake/modules/version-helper.cmake
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generates TypeART meta data API."
)

set(RUNTIME_LIB_SOURCES
    AccessCounter.h
    CallbackInterface.h
    RuntimeData.h
    RuntimeInterface.h
    TypeResolution.cpp
    AllocationTracking.cpp
    AllocationTracking.h
    TypeResolution.h
    Runtime.cpp
    Runtime.h
    ${TYPEART_META_SOURCE}
    $<$<BOOL:${TYPEART_MPI_LOGGER}>:../support/MPILogger.cpp>
)

add_library(${TYPEART_PREFIX}_Runtime SHARED ${RUNTIME_LIB_SOURCES})
set_target_properties(
  ${TYPEART_PREFIX}_Runtime
  PROPERTIES
  OUTPUT_NAME "${PROJECT_NAME}Runtime"
  EXPORT_NAME "Runtime"
)
add_library(typeart::Runtime ALIAS ${TYPEART_PREFIX}_Runtime)

target_link_libraries(
  ${TYPEART_PREFIX}_Runtime
  PRIVATE typeart::TypesStatic
          typeart::SystemStatic
          LLVMCore
          LLVMSupport
          $<$<BOOL:${TYPEART_MPI_LOGGER}>:MPI::MPI_CXX>
          $<$<BOOL:${TYPEART_PHMAP}>:phpmap::phpmap>
          $<$<BOOL:${TYPEART_ABSEIL}>:absl::btree>
          $<$<BOOL:${TYPEART_SAFEPTR}>:sf::pointer>
          $<$<NOT:$<BOOL:${TYPEART_DISABLE_THREAD_SAFETY}>>:Threads::Threads>
)

target_include_directories(
  ${TYPEART_PREFIX}_Runtime ${warning_guard}
  PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>
         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib/runtime>
         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib/typelib>
  PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib/passes/support>
          $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
)

target_include_directories(${TYPEART_PREFIX}_Runtime SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})

target_compile_definitions(
  ${TYPEART_PREFIX}_Runtime
  PRIVATE TYPEART_LOG_LEVEL=${TYPEART_LOG_LEVEL_RT}
          $<$<BOOL:${TYPEART_MPI_LOGGER}>:TYPEART_MPI_LOGGER=1>
          $<$<BOOL:${TYPEART_SOFTCOUNTERS}>:ENABLE_SOFTCOUNTER=1>
          $<$<BOOL:${TYPEART_PHMAP}>:TYPEART_PHMAP>
          $<$<BOOL:${TYPEART_ABSEIL}>:TYPEART_ABSEIL>
          $<$<BOOL:${TYPEART_SAFEPTR}>:USE_SAFEPTR>
          $<$<BOOL:${TYPEART_DISABLE_THREAD_SAFETY}>:TYPEART_DISABLE_THREAD_SAFETY>
)

typeart_target_compile_options(${TYPEART_PREFIX}_Runtime)
typeart_target_define_file_basename(${TYPEART_PREFIX}_Runtime)
typeart_target_coverage_options(${TYPEART_PREFIX}_Runtime)

if(TYPEART_TSAN)
  typeart_target_tsan_options(${TYPEART_PREFIX}_Runtime)
endif()

if(TYPEART_ASAN)
  typeart_target_asan_options(${TYPEART_PREFIX}_Runtime)
endif()

if(TYPEART_UBSAN)
  typeart_target_ubsan_options(${TYPEART_PREFIX}_Runtime)
endif()

make_tidy_check(${TYPEART_PREFIX}_Runtime "${RUNTIME_LIB_SOURCES}")

set(CONFIG_NAME ${PROJECT_NAME}Runtime)
set(TARGETS_EXPORT_NAME ${CONFIG_NAME}Targets)

install(FILES RuntimeInterface.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)

install(
  TARGETS ${TYPEART_PREFIX}_Runtime
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

#configure_package_config_file(
#  ${PROJECT_SOURCE_DIR}/cmake/Config.cmake.in
#  ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}Config.cmake
#  INSTALL_DESTINATION ${TYPEART_INSTALL_CONFIGDIR}
#)
#
#install(FILES
#  ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}Config.cmake
#  DESTINATION ${TYPEART_INSTALL_CONFIGDIR}
#)

set(PROJECT_NAME typeart-rt)
set(TARGETS_EXPORT_NAME ${PROJECT_NAME}-targets)

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
  $<$<BOOL:${MPI_LOGGER}>:../support/MPILogger.cpp>
)

add_library(typeart-rt SHARED
  ${RUNTIME_LIB_SOURCES}
)
add_library(typeart::typeart-rt ALIAS typeart-rt)

target_link_libraries(typeart-rt
  PRIVATE
    typeart::typelib
    LLVMCore
    LLVMSupport
    $<$<BOOL:${MPI_LOGGER}>:MPI::MPI_CXX>
    $<$<BOOL:${USE_BTREE}>:google::btree>
    $<$<BOOL:${USE_ABSL}>:absl::btree>
    $<$<BOOL:${ENABLE_SAFEPTR}>:sf::pointer>
)

target_include_directories(typeart-rt
  PUBLIC
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib/runtime>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib/typelib>
  PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib/passes/support>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
  )

target_include_directories(typeart-rt
  SYSTEM
  PRIVATE
  ${LLVM_INCLUDE_DIRS}
  )


target_compile_definitions(typeart-rt
  PRIVATE
    LOG_LEVEL=${LOG_LEVEL_RT}
    $<$<BOOL:${MPI_LOGGER}>:MPI_LOGGER=1>
    $<$<BOOL:${SOFTCOUNTERS}>:ENABLE_SOFTCOUNTER=1>
    $<$<BOOL:${USE_BTREE}>:USE_BTREE>
    $<$<BOOL:${USE_ABSL}>:USE_ABSL>
    $<$<BOOL:${ENABLE_SAFEPTR}>:USE_SAFEPTR>
    $<$<BOOL:${DISABLE_THREAD_SAFETY}>:DISABLE_THREAD_SAFETY>
)

target_project_compile_definitions(typeart-rt)
target_define_file_basename(typeart-rt)
target_project_coverage_options(typeart-rt)

if(ENABLE_TSAN)
  target_tsan_options(typeart-rt)
endif()

if(ENABLE_ASAN)
  target_asan_options(typeart-rt)
endif()

if(ENABLE_UBSAN)
  target_ubsan_options(typeart-rt)
endif()

make_tidy_check(typeart-rt
  "${RUNTIME_LIB_SOURCES}"
)

install(
  TARGETS typeart-rt
  EXPORT ${TARGETS_EXPORT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
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

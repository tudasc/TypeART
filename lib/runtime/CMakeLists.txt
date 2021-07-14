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

add_library(Runtime SHARED
  ${RUNTIME_LIB_SOURCES}
)
set_property(TARGET Runtime PROPERTY OUTPUT_NAME "${PROJECT_NAME}Runtime")
add_library(typeart::Runtime ALIAS Runtime)

target_link_libraries(Runtime
  PRIVATE
    typeart::Types
    LLVMCore
    LLVMSupport
    $<$<BOOL:${MPI_LOGGER}>:MPI::MPI_CXX>
    $<$<BOOL:${USE_BTREE}>:google::btree>
    $<$<BOOL:${USE_ABSL}>:absl::btree>
    $<$<BOOL:${ENABLE_SAFEPTR}>:sf::pointer>
)

target_include_directories(Runtime
  PUBLIC
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib/runtime>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib/typelib>
  PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib/passes/support>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
  )

target_include_directories(Runtime
  SYSTEM
  PRIVATE
  ${LLVM_INCLUDE_DIRS}
  )


target_compile_definitions(Runtime
  PRIVATE
    LOG_LEVEL=${LOG_LEVEL_RT}
    $<$<BOOL:${MPI_LOGGER}>:MPI_LOGGER=1>
    $<$<BOOL:${SOFTCOUNTERS}>:ENABLE_SOFTCOUNTER=1>
    $<$<BOOL:${USE_BTREE}>:USE_BTREE>
    $<$<BOOL:${USE_ABSL}>:USE_ABSL>
    $<$<BOOL:${ENABLE_SAFEPTR}>:USE_SAFEPTR>
    $<$<BOOL:${DISABLE_THREAD_SAFETY}>:DISABLE_THREAD_SAFETY>
)

target_project_compile_definitions(Runtime)
target_define_file_basename(Runtime)
target_project_coverage_options(Runtime)

if(ENABLE_TSAN)
  target_tsan_options(Runtime)
endif()

if(ENABLE_ASAN)
  target_asan_options(Runtime)
endif()

if(ENABLE_UBSAN)
  target_ubsan_options(Runtime)
endif()

make_tidy_check(Runtime
  "${RUNTIME_LIB_SOURCES}"
)

install(
  TARGETS Runtime
  EXPORT ${TARGETS_EXPORT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
  FILES RuntimeInterface.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)

include(CMakeDependentOption)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

find_package(LLVM 10 REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

find_package(OpenMP QUIET)

set(THREADS_PREFER_PTHREAD_FLAG 1)
set(CMAKE_THREAD_PREFER_PTHREAD 1)
find_package(Threads QUIET)

set(LOG_LEVEL 0 CACHE STRING "Granularity of LLVM pass logger. 3 ist most verbose, 0 is least.")
set(LOG_LEVEL_RT 0 CACHE STRING "Granularity of runtime logger. 3 ist most verbose, 0 is least.")
option(SHOW_STATS "Passes show the statistics vars." ON)
option(MPI_LOGGER "Whether the logger should use MPI." ON)
option(MPI_INTERCEPT_LIB "Build MPI interceptor library for prototyping and testing." ON)
option(SOFTCOUNTERS "Enable software tracking of #tracked addrs. / #distinct checks / etc." OFF)
option(TEST_CONFIG "Set logging levels to appropriate levels for test runner to succeed" OFF)
option(ENABLE_CODE_COVERAGE "Enable code coverage statistics" OFF)
option(ENABLE_LLVM_CODE_COVERAGE "Enable llvm-cov code coverage statistics" OFF)
option(TEST_CONFIGURE_IDE "Add targets so the IDE (e.g., Clion) can interpret test files better" ON)
mark_as_advanced(TEST_CONFIGURE_IDE)
option(ENABLE_TSAN "Build runtime lib and tests with fsanitize=thread" OFF)
option(ENABLE_SAFEPTR "Use external safe_ptr map wrapper instead of mutex" OFF)
cmake_dependent_option(DISABLE_THREAD_SAFETY "Explicitly make runtime *not* thread-safe." OFF
  "NOT ENABLE_SAFEPTR" OFF
)
cmake_dependent_option(ENABLE_ASAN "Build runtime lib and tests with fsanitize=address." OFF
  "NOT ENABLE_TSAN" OFF
)
cmake_dependent_option(ENABLE_UBSAN "Build runtime lib and tests with fsanitize=undefined." OFF
  "NOT ENABLE_TSAN" OFF
)
option(INSTALL_UTIL_SCRIPTS "Install single file build and run scripts" OFF)
mark_as_advanced(INSTALL_UTIL_SCRIPTS)
option(ENABLE_MPI_WRAPPER "Generate mpicc and mpic++ wrapper for TypeART" ON)
option(USE_ABSL "Enable usage of abseil's btree-backed map instead of std::map for the runtime." ON)
cmake_dependent_option(USE_BTREE "Enable usage of btree-backed map instead of std::map for the runtime." ON
  "NOT USE_ABSL" OFF
)

include(AddLLVM)
include(llvm-lit)
include(clang-tidy)
include(clang-format)
include(llvm-util)
include(log-util)
include(coverage)
include(sanitizer-targets)
include(target-util)

if(TEST_CONFIG)
  set(LOG_LEVEL 2 CACHE STRING "" FORCE)
  set(LOG_LEVEL_RT 3 CACHE STRING "" FORCE)
endif()

if(MPI_LOGGER
   OR ENABLE_MPI_WRAPPER
   OR MPI_INTERCEPT_LIB
)
  find_package(MPI REQUIRED)
endif()

if(MPI_INTERCEPT_LIB)
  find_package(PythonInterp REQUIRED)
endif()

if(NOT CMAKE_BUILD_TYPE)
  # set default build type
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
  message(STATUS "Building as debug (default)")
endif()

if(NOT CMAKE_DEBUG_POSTFIX AND CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_DEBUG_POSTFIX "-d")
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  # set default install path
  set(CMAKE_INSTALL_PREFIX
      "${typeart_SOURCE_DIR}/install/typeart"
      CACHE PATH "Default install path" FORCE
  )
  message(STATUS "Installing to (default): ${CMAKE_INSTALL_PREFIX}")
endif()

set(TARGETS_EXPORT_NAME ${PROJECT_NAME}Targets)
set(TYPEART_INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

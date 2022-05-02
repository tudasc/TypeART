include(CMakeDependentOption)
include(CMakePackageConfigHelpers)
include(FeatureSummary)
include(CheckCSourceCompiles)

find_package(LLVM CONFIG HINTS "${LLVM_DIR}")
if(NOT LLVM_FOUND)
  message(STATUS "LLVM not found at: ${LLVM_DIR}.")
  find_package(LLVM REQUIRED CONFIG)
endif()
set_package_properties(LLVM PROPERTIES
  URL https://llvm.org/
  TYPE REQUIRED
  PURPOSE
  "LLVM framework installation required to compile (and apply) TypeART."
)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

find_package(OpenMP QUIET)
set_package_properties(OpenMP PROPERTIES
  TYPE OPTIONAL
  PURPOSE
  "OpenMP is optionally used by the test suite to verify that the LLVM passes handle OpenMPk codes."
)

set(TYPEART_LOG_LEVEL 0 CACHE STRING "Granularity of LLVM pass logger. 3 is most verbose, 0 is least.")
set(TYPEART_LOG_LEVEL_RT 0 CACHE STRING "Granularity of runtime logger. 3 is most verbose, 0 is least.")

option(TYPEART_SHOW_STATS "Passes show the statistics vars." ON)
add_feature_info(SHOW_STATS TYPEART_SHOW_STATS "Show compile time statistics of TypeART's LLVM passes.")

option(TYPEART_MPI_LOGGER "Whether the logger should use MPI." ON)
add_feature_info(MPI_LOGGER TYPEART_MPI_LOGGER "Logger supports MPI context.")

option(TYPEART_MPI_INTERCEPT_LIB "Build MPI interceptor library for prototyping and testing." ON)
add_feature_info(MPI_INTERCEPT_LIB TYPEART_MPI_INTERCEPT_LIB "Build TypeART's MPI tool, which intercepts MPI calls and applies buffer checks.")

option(TYPEART_SOFTCOUNTERS "Enable software tracking of #tracked addrs. / #distinct checks / etc." OFF)
add_feature_info(SOFTCOUNTER TYPEART_SOFTCOUNTERS "Runtime collects various counters of memory ops/check operations.")

option(TYPEART_TEST_CONFIG "Set logging levels to appropriate levels for test runner to succeed" OFF)
add_feature_info(TEST_CONFIG TYPEART_TEST_CONFIG "Test config to support lit test suite with appropriate diagnostic logging levels.")

option(TYPEART_CODE_COVERAGE "Enable code coverage statistics" OFF)
add_feature_info(CODE_COVERAGE TYPEART_CODE_COVERAGE "Enable code coverage with lcov.")

option(TYPEART_LLVM_CODE_COVERAGE "Enable llvm-cov code coverage statistics" OFF)
add_feature_info(LLVM_CODE_COVERAGE TYPEART_LLVM_CODE_COVERAGE "Enable LLVM code coverage with llvm-cov.")

option(TYPEART_SAFEPTR "Use external safe_ptr map wrapper instead of mutex" OFF)
add_feature_info(SAFEPTR TYPEART_SAFEPTR "External library object_threadsafe provides lock-free runtime pointer map wrapper.")

cmake_dependent_option(TYPEART_DISABLE_THREAD_SAFETY "Explicitly make runtime *not* thread-safe." OFF
  "NOT TYPEART_SAFEPTR" OFF
)
add_feature_info(DISABLE_THREAD_SAFETY TYPEART_DISABLE_THREAD_SAFETY "Thread-safety features of runtime disabled.")

option(TYPEART_TSAN "Build runtime lib and tests with fsanitize=thread" OFF)
add_feature_info(TSAN TYPEART_TSAN "Build with sanitizer \"tsan\".")

cmake_dependent_option(TYPEART_ASAN "Build runtime lib and tests with fsanitize=address." OFF
  "NOT TYPEART_TSAN" OFF
)
add_feature_info(ASAN TYPEART_ASAN "Build with sanitizer \"asan\".")

cmake_dependent_option(TYPEART_UBSAN "Build runtime lib and tests with fsanitize=undefined." OFF
  "NOT TYPEART_TSAN" OFF
)
add_feature_info(UBSAN TYPEART_UBSAN "Build with sanitizer \"ubsan=undefined\".")

option(TYPEART_MPI_WRAPPER "Generate mpicc and mpic++ wrapper for TypeART" ON)
add_feature_info(MPI_WRAPPER TYPEART_MPI_WRAPPER "Generate TypeART compiler wrapper for mpicc and mpic++.")

option(TYPEART_ABSEIL "Enable usage of Abseil's btree-backed map instead of std::map for the runtime." ON)
add_feature_info(ABSEIL TYPEART_ABSEIL "External library \"Abseil\" replaces runtime std::map with btree-backed map.")

cmake_dependent_option(TYPEART_PHMAP "Enable usage of project \"phmap\" btree-backed map for the runtime." ON
  "NOT TYPEART_ABSEIL" OFF
)
add_feature_info(PHMAP TYPEART_PHMAP "External library \"parallel-hashmap\" replaces runtime std::map with btree-backed map.")

option(TYPEART_INSTALL_UTIL_SCRIPTS "Install single file build and run scripts" OFF)
mark_as_advanced(TYPEART_INSTALL_UTIL_SCRIPTS)

option(TYPEART_TEST_CONFIGURE_IDE "Add targets so the IDE (e.g., Clion) can interpret test files better" ON)
mark_as_advanced(TYPEART_TEST_CONFIGURE_IDE)

option(TYPEART_CONFIG_DIR_IS_SHARE "Install to \"share/cmake/\" instead of \"lib/cmake/\"" OFF)
mark_as_advanced(TYPEART_CONFIG_DIR_IS_SHARE)

set(warning_guard "")
if(NOT TYPEART_IS_TOP_LEVEL)
  option(
      TYPEART_INCLUDES_WITH_SYSTEM
      "Use SYSTEM modifier for TypeART includes to disable warnings."
      ON
  )
  mark_as_advanced(TYPEART_INCLUDES_WITH_SYSTEM)

  if(TYPEART_INCLUDES_WITH_SYSTEM)
    set(warning_guard SYSTEM)
  endif()
endif()

include(AddLLVM)
include(modules/clang-tidy)
include(modules/clang-format)
include(modules/llvm-util)
include(modules/log-util)
include(modules/coverage)
include(modules/sanitizer-targets)
include(modules/target-util)

if(TYPEART_TEST_CONFIG)
  set(TYPEART_LOG_LEVEL 2 CACHE STRING "" FORCE)
  set(TYPEART_LOG_LEVEL_RT 3 CACHE STRING "" FORCE)
endif()

set(THREADS_PREFER_PTHREAD_FLAG 1)
set(CMAKE_THREAD_PREFER_PTHREAD 1)
if(NOT TYPEART_DISABLE_THREAD_SAFETY)
  find_package(Threads REQUIRED)
else()
  find_package(Threads QUIET)
endif()
set_package_properties(Threads PROPERTIES
  TYPE RECOMMENDED
  PURPOSE
  "Threads are needed to compile our thread-safe typeart::Runtime due to use of std::mutex etc."
)

if(TYPEART_MPI_LOGGER
   OR TYPEART_MPI_WRAPPER
   OR TYPEART_MPI_INTERCEPT_LIB
)
  find_package(MPI REQUIRED)
endif()
set_package_properties(MPI PROPERTIES
  TYPE RECOMMENDED
  PURPOSE
  "The MPI library is needed for several TypeART components: MPI logging, MPI compiler wrapper, and the MPI interceptor tool."
)

if(TYPEART_MPI_INTERCEPT_LIB)
  find_package(Python3 REQUIRED)
endif()
set_package_properties(Python3 PROPERTIES
  TYPE RECOMMENDED
  PURPOSE
  "The Python3 interpreter is used for lit-testing and the MPI interceptor tool code generation."
)

typeart_find_llvm_progs(TYPEART_CLANG_EXEC "clang-${LLVM_VERSION_MAJOR};clang" DEFAULT_EXE "clang")
typeart_find_llvm_progs(TYPEART_CLANGCXX_EXEC "clang++-${LLVM_VERSION_MAJOR};clang++" DEFAULT_EXE "clang++")
typeart_find_llvm_progs(TYPEART_LLC_EXEC "llc-${LLVM_VERSION_MAJOR};llc" DEFAULT_EXE "llc")
typeart_find_llvm_progs(TYPEART_OPT_EXEC "opt-${LLVM_VERSION_MAJOR};opt" DEFAULT_EXE "opt")

if(TYPEART_IS_TOP_LEVEL)
  if(NOT CMAKE_BUILD_TYPE)
    # set default build type
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
    message(STATUS "Building as debug (default)")
  endif()

  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    # set default install path
    set(CMAKE_INSTALL_PREFIX
        "${typeart_SOURCE_DIR}/install/typeart"
        CACHE PATH "Default install path" FORCE
    )
    message(STATUS "Installing to (default): ${CMAKE_INSTALL_PREFIX}")
  endif()

  # TYPEART_DEBUG_POSTFIX is only used for Config
  if(CMAKE_DEBUG_POSTFIX)
    set(TYPEART_DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
  else()
    set(TYPEART_DEBUG_POSTFIX "-d")
  endif()

  # CMAKE_DEBUG_POSTFIX is used for targets
  if(NOT CMAKE_DEBUG_POSTFIX AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_DEBUG_POSTFIX ${TYPEART_DEBUG_POSTFIX})
  endif()
else()
  set(TYPEART_DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
endif()

include(GNUInstallDirs)

set(TYPEART_PREFIX ${PROJECT_NAME})
set(TARGETS_EXPORT_NAME ${TYPEART_PREFIX}Targets)

if(TYPEART_CONFIG_DIR_IS_SHARE)
  set(TYPEART_INSTALL_CONFIGDIR ${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME})
else()
  set(TYPEART_INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
endif()

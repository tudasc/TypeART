include(CMakeDependentOption)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
include(FeatureSummary)

find_package(LLVM CONFIG HINTS "${LLVM_DIR}" NO_DEFAULT_PATH)
if(NOT LLVM_FOUND)
  message(STATUS "LLVM not found at: ${LLVM_DIR}.")
  find_package(LLVM 10 REQUIRED CONFIG)
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

set(LOG_LEVEL 0 CACHE STRING "Granularity of LLVM pass logger. 3 ist most verbose, 0 is least.")
set(LOG_LEVEL_RT 0 CACHE STRING "Granularity of runtime logger. 3 ist most verbose, 0 is least.")

option(SHOW_STATS "Passes show the statistics vars." ON)
add_feature_info(SHOW_STATS SHOW_STATS "Show compile time statistics of TypeART's LLVM passes.")

option(MPI_LOGGER "Whether the logger should use MPI." ON)
add_feature_info(MPI_LOGGER MPI_LOGGER "Logger supports MPI context.")

option(MPI_INTERCEPT_LIB "Build MPI interceptor library for prototyping and testing." ON)
add_feature_info(MPI_INTERCEPT_LIB MPI_INTERCEPT_LIB "Build TypeART's MPI tool, which intercepts MPI calls and applies buffer checks.")

option(SOFTCOUNTERS "Enable software tracking of #tracked addrs. / #distinct checks / etc." OFF)
add_feature_info(SOFTCOUNTER SOFTCOUNTERS "Runtime collects various counters of memory ops/check operations.")

option(TEST_CONFIG "Set logging levels to appropriate levels for test runner to succeed" OFF)
add_feature_info(TEST_CONFIG TEST_CONFIG "Test config to support lit test suite with appropriate diagnostic logging levels.")

option(ENABLE_CODE_COVERAGE "Enable code coverage statistics" OFF)
add_feature_info(CODE_COVERAGE ENABLE_CODE_COVERAGE "Enable code coverage with lcov.")

option(ENABLE_LLVM_CODE_COVERAGE "Enable llvm-cov code coverage statistics" OFF)
add_feature_info(LLVM_CODE_COVERAGE ENABLE_LLVM_CODE_COVERAGE "Enable LLVM code coverage with llvm-cov.")

option(ENABLE_SAFEPTR "Use external safe_ptr map wrapper instead of mutex" OFF)
add_feature_info(SAFEPTR ENABLE_SAFEPTR "External library object_threadsafe provides lock-free runtime pointer map wrapper.")

cmake_dependent_option(DISABLE_THREAD_SAFETY "Explicitly make runtime *not* thread-safe." OFF
  "NOT ENABLE_SAFEPTR" OFF
)
add_feature_info(DISABLE_THREAD_SAFETY DISABLE_THREAD_SAFETY "Thread-safety features of runtime disabled.")

option(ENABLE_TSAN "Build runtime lib and tests with fsanitize=thread" OFF)
add_feature_info(TSAN ENABLE_TSAN "Build with sanitizer \"tsan\".")

cmake_dependent_option(ENABLE_ASAN "Build runtime lib and tests with fsanitize=address." OFF
  "NOT ENABLE_TSAN" OFF
)
add_feature_info(ASAN ENABLE_ASAN "Build with sanitizer \"asan\".")

cmake_dependent_option(ENABLE_UBSAN "Build runtime lib and tests with fsanitize=undefined." OFF
  "NOT ENABLE_TSAN" OFF
)
add_feature_info(UBSAN ENABLE_UBSAN "Build with sanitizer \"ubsan=undefined\".")

option(ENABLE_MPI_WRAPPER "Generate mpicc and mpic++ wrapper for TypeART" ON)
add_feature_info(MPI_WRAPPER ENABLE_MPI_WRAPPER "Generate TypeART compiler wrapper for mpicc and mpic++.")

option(USE_ABSL "Enable usage of abseil's btree-backed map instead of std::map for the runtime." ON)
add_feature_info(ABSL USE_ABSL "External library \"Abseil\" replaces runtime std::map with btree-backed map.")

cmake_dependent_option(USE_BTREE "Enable usage of btree-backed map instead of std::map for the runtime." ON
  "NOT USE_ABSL" OFF
)
add_feature_info(BTREE USE_BTREE "*Deprecated* External library replaces runtime std::map with btree-backed map.")

option(INSTALL_UTIL_SCRIPTS "Install single file build and run scripts" OFF)
mark_as_advanced(INSTALL_UTIL_SCRIPTS)

option(TEST_CONFIGURE_IDE "Add targets so the IDE (e.g., Clion) can interpret test files better" ON)
mark_as_advanced(TEST_CONFIGURE_IDE)

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

set(THREADS_PREFER_PTHREAD_FLAG 1)
set(CMAKE_THREAD_PREFER_PTHREAD 1)
if(NOT DISABLE_THREAD_SAFETY)
  find_package(Threads REQUIRED)
else()
  find_package(Threads QUIET)
endif()
set_package_properties(Threads PROPERTIES
  TYPE RECOMMENDED
  PURPOSE
  "Threads are needed to compile our thread-safe typeart::Runtime due to use of std::mutex etc."
)

if(MPI_LOGGER
   OR ENABLE_MPI_WRAPPER
   OR MPI_INTERCEPT_LIB
)
  find_package(MPI REQUIRED)
endif()
set_package_properties(MPI PROPERTIES
  TYPE RECOMMENDED
  PURPOSE
  "The MPI library is needed for several TypeART components: MPI logging, MPI compiler wrapper, and the MPI interceptor tool."
)

if(MPI_INTERCEPT_LIB)
  find_package(PythonInterp REQUIRED)
endif()
set_package_properties(PythonInterp PROPERTIES
  TYPE RECOMMENDED
  PURPOSE
  "The Python interp is used for lit-testing and the MPI interceptor tool code generation."
)

typeart_find_llvm_progs(TYPEART_CLANG_EXEC "clang;clang-13;clang-12;clang-11;clang-10" "clang")
typeart_find_llvm_progs(TYPEART_CLANGCXX_EXEC "clang++;clang-13;clang-12;clang-11;clang++-10" "clang++")
typeart_find_llvm_progs(TYPEART_LLC_EXEC "llc;llc-13;llc-12;llc-11;llc-10" "llc")
typeart_find_llvm_progs(TYPEART_OPT_EXEC "opt;opt-13;opt-12;opt-11;opt-10" "opt")
typeart_find_llvm_progs(TYPEART_FILECHECK_EXEC "FileCheck;FileCheck-13;FileCheck-12;FileCheck-11;FileCheck-10" "FileCheck")

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

set(TYPEART_PREFIX ${PROJECT_NAME})
set(TARGETS_EXPORT_NAME ${TYPEART_PREFIX}Targets)
set(TYPEART_INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

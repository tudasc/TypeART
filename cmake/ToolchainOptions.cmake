find_package(LLVM 10 REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

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

# tycart options - begin
option(WITH_FTI "Enable FTI as backend for TyCart" OFF)
option(WITH_VELOC "Enable VeloC as backend for TyCart" OFF)
option(WITH_MINI_CPR "Enable mini-cpr as backend for TyCart" OFF)
set(FTI_INSTALL_DIR "" CACHE PATH "Path to FTI install directory")
set(VELOC_INSTALL_DIR "" CACHE PATH "Path to VeloC install directory")
set(MINI_INSTALL_DIR "" CACHE PATH "Path to mini-cpr install directory")
# tycart options - end

include(AddLLVM)
include(llvm-lit)
include(clang-tidy)
include(clang-format)
include(llvm-util)
include(log-util)
include(coverage)

if (TEST_CONFIG)
  set(LOG_LEVEL 2 CACHE STRING "" FORCE)
  set(LOG_LEVEL_RT 3 CACHE STRING "" FORCE)
endif ()

if (MPI_LOGGER)
  find_package(MPI REQUIRED)
endif ()

if (NOT CMAKE_BUILD_TYPE)
  # set default build type
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
  message(STATUS "Building as debug (default)")
endif ()

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  # set default install path
  set(CMAKE_INSTALL_PREFIX "${typeart_SOURCE_DIR}/install/typeart" CACHE PATH "Default install path" FORCE)
  message(STATUS "Installing to (default): ${CMAKE_INSTALL_PREFIX}")
endif ()

function(target_project_compile_options target)
  cmake_parse_arguments(ARG "" "" "PRIVATE_FLAGS;PUBLIC_FLAGS" ${ARGN})

  target_compile_options(${target} PRIVATE
    -Wall -Wextra -pedantic
    -Wunreachable-code -Wwrite-strings
    -Wpointer-arith -Wcast-align
    -Wcast-qual -Wno-unused-parameter
    )

  if (ARG_PRIVATE_FLAGS)
    target_compile_options(${target} PRIVATE
      "${ARG_PRIVATE_FLAGS}"
      )
  endif ()

  if (ARG_PUBLIC_FLAGS)
    target_compile_options(${target} PUBLIC
      "${ARG_PUBLIC_FLAGS}"
      )
  endif ()
endfunction()

function(target_project_compile_definitions target)
  cmake_parse_arguments(ARG "" "" "PRIVATE_DEFS;PUBLIC_DEFS" ${ARGN})

  if (ARG_PRIVATE_DEFS)
    target_compile_definitions(${target} PRIVATE
      "${ARG_PRIVATE_DEFS}"
      )
  endif ()

  if (ARG_PUBLIC_DEFS)
    target_compile_definitions(${target} PUBLIC
      "${ARG_PUBLIC_DEFS}"
      )
  endif ()
endfunction()

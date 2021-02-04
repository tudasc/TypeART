cmake_minimum_required(VERSION 3.14)

project(typeart
  VERSION 1.5
)

set(TYPEART_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH 
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

include(ToolchainOptions)
include(CMakePackageConfigHelpers)

add_format_target(format-sources
  "Formats project source files"
  TARGETS
  lib/passes/*.cpp
  lib/passes/*.h
  lib/runtime/*.cpp
  lib/runtime/*.h
  lib/typelib/*.cpp
  lib/typelib/*.h
  lib/support/*.cpp
  lib/support/*.h
  )

add_subdirectory(externals)
get_directory_property(safe_ptr_include_dir DIRECTORY externals/safe_ptr DEFINITION safe_ptr_include_dir)

add_subdirectory(lib)
add_subdirectory(scripts)
enable_testing()
add_subdirectory(test)

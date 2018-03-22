cmake_minimum_required(VERSION 3.5)

project(llvm_must_support)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 14)
list(APPEND CMAKE_MODULE_PATH 
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

include(ToolchainOptions)

add_format_target(format-sources
  "Formats project source files"
  TARGETS lib/*.cpp 
          lib/*.h
          runtime/*.cpp
          runtime/*.h
          configio/*.cpp
          configio/*.h
)

add_subdirectory(lib)
add_subdirectory(configio)
add_subdirectory(runtime)
enable_testing()
add_subdirectory(test)
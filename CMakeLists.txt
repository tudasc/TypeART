cmake_minimum_required(VERSION 3.5)

project(llvm_must_support)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 14)

if (UNIX)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Wunreachable-code -Wwrite-strings -Wpointer-arith -Wcast-align -Wcast-qual -Wno-unused-parameter")
endif (UNIX)

include(${PROJECT_SOURCE_DIR}/cmake/ToolchainOptions.cmake)

find_package(LLVM REQUIRED CONFIG)

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(lib)
add_subdirectory(runtime)

add_format_target(format-all
  "Formats project source files"
)
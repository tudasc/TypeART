cmake_minimum_required(VERSION 3.4.3)

project(llvm_must_support)

find_package(LLVM REQUIRED CONFIG)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${LLVM_DEFINITIONS})

add_subdirectory(lib)
add_subdirectory(runtime)

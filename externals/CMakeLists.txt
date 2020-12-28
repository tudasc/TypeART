include(CMakeDependentOption)

set(EXTERNAL_DIR ${CMAKE_CURRENT_BINARY_DIR})
# TODO may remove later
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE STRING "" FORCE)

option(USE_BTREE "Enable usage of btree-backed map instead of std::map for the runtime." OFF)
CMAKE_DEPENDENT_OPTION(USE_ABSL "Enable usage of abseil's btree-backed map instead of std::map for the runtime." ON "NOT USE_BTREE" ON)

if(USE_BTREE)
  add_subdirectory(btree)
endif()

if(USE_ABSL)
  add_subdirectory(abseil)
endif()
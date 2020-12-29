include(CMakeDependentOption)

set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE STRING "" FORCE)
include(FetchContent)

mark_as_advanced(
  FETCHCONTENT_BASE_DIR
  FETCHCONTENT_QUIET
  FETCHCONTENT_FULLY_DISCONNECTED
)

option(USE_ABSL "Enable usage of abseil's btree-backed map instead of std::map for the runtime." ON )
CMAKE_DEPENDENT_OPTION(USE_BTREE "Enable usage of btree-backed map instead of std::map for the runtime." ON "NOT USE_ABSL" ON)

if(USE_BTREE)
  add_subdirectory(btree)
endif()

if(USE_ABSL)
  add_subdirectory(abseil)
endif()
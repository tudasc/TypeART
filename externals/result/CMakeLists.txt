FetchContent_Declare(
  result
  GIT_REPOSITORY https://github.com/bitwizeshift/result
  GIT_SHALLOW 1
  GIT_TAG v1.0.0
)

FetchContent_GetProperties(result)

if(NOT result_POPULATED)
  FetchContent_Populate(result)

  add_library(result INTERFACE)
  target_include_directories(result SYSTEM INTERFACE ${result_SOURCE_DIR}/include)
  add_library(result::result ALIAS result)
endif()

mark_as_advanced(
  FETCHCONTENT_SOURCE_DIR_RESULT
  FETCHCONTENT_UPDATES_DISCONNECTED_RESULT
)

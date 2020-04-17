include(FetchContent)
FetchContent_Declare(
  cpp-abseil
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
)

FetchContent_GetProperties(cpp-abseil)
if(NOT cpp-abseil_POPULATED)
  FetchContent_Populate(cpp-abseil)
  add_subdirectory(${cpp-abseil_SOURCE_DIR} ${cpp-abseil_BINARY_DIR})
#  set_target_properties(absl::btree PROPERTIES IMPORTED_GLOBAL TRUE)
endif()

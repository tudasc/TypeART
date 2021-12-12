set(CMAKE_POSITION_INDEPENDENT_CODE ON)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_SHALLOW 1
  GIT_TAG 8.0.1
)

FetchContent_MakeAvailable(fmt)

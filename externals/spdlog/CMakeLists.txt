set(CMAKE_POSITION_INDEPENDENT_CODE ON)

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog
  GIT_SHALLOW 1
  GIT_TAG v1.9.2
)

FetchContent_MakeAvailable(spdlog)

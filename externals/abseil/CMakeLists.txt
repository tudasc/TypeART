include(FetchContent)

FetchContent_Declare(
  cpp-abseil
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
)

FetchContent_MakeAvailable(cpp-abseil)

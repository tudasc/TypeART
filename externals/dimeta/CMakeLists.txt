FetchContent_Declare(
  llvm-dimeta
  GIT_REPOSITORY https://github.com/ahueck/llvm-dimeta
  GIT_TAG v0.1.0
  GIT_SHALLOW 1
)

FetchContent_MakeAvailable(llvm-dimeta)

mark_as_advanced(
  DIMETA_USE_HEAPALLOCSITE
  DIMETA_ENABLE_COVERAGE)
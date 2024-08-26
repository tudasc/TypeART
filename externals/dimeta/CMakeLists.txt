FetchContent_Declare(
  llvm-ditype
  GIT_REPOSITORY https://github.com/ahueck/llvm-di-type
  GIT_TAG feat/typeart
  GIT_SHALLOW 1
)

FetchContent_MakeAvailable(llvm-ditype)

mark_as_advanced(
  DIMETA_USE_HEAPALLOCSITE
  DIMETA_ENABLE_COVERAGE)
FetchContent_Declare(
  llvm-dimeta
  GIT_REPOSITORY https://github.com/ahueck/llvm-dimeta
  GIT_TAG v0.2.0
  GIT_SHALLOW 1
)

#FetchContent_MakeAvailable(llvm-dimeta)

# TODO need exclude to not install llvm-dimeta with TypeART
FetchContent_GetProperties(llvm-dimeta)
if(NOT llvm-dimeta_POPULATED)
  FetchContent_Populate(llvm-dimeta)
  add_subdirectory(${llvm-dimeta_SOURCE_DIR} ${llvm-dimeta_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

mark_as_advanced(
  DIMETA_USE_HEAPALLOCSITE
  DIMETA_ENABLE_COVERAGE
  DIMETA_TEST_CONFIG
  DIMETA_LOG_LEVEL
  FETCHCONTENT_UPDATES_DISCONNECTED_LLVM-DIMETA
  FETCHCONTENT_SOURCE_DIR_LLVM-DIMETA
)

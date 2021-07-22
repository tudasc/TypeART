FetchContent_Declare(
  safe_ptr
  GIT_REPOSITORY https://github.com/AlexeyAB/object_threadsafe.git
  GIT_SHALLOW 1
)

FetchContent_GetProperties(safe_ptr)

if(NOT safe_ptr_POPULATED)
  FetchContent_Populate(safe_ptr)

  add_library(safeptr INTERFACE IMPORTED GLOBAL)
  target_include_directories(safeptr SYSTEM INTERFACE ${safe_ptr_SOURCE_DIR})
  #target_compile_features(safeptr INTERFACE cxx_std_17 )
  add_library(sf::pointer ALIAS safeptr)
endif()

mark_as_advanced(
  FETCHCONTENT_SOURCE_DIR_SAFE_PTR
  FETCHCONTENT_UPDATES_DISCONNECTED_SAFE_PTR
)

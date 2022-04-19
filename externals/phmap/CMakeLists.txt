FetchContent_Declare(
  cpp-phmap
  GIT_REPOSITORY https://github.com/greg7mdp/parallel-hashmap.git
  GIT_SHALLOW 1
  GIT_TAG 48e86db1f4ebc85145d384ba061c4f615cb3106f
)

FetchContent_GetProperties(cpp-phmap)

if(NOT cpp-phmap_POPULATED)
  FetchContent_Populate(cpp-phmap)

  add_library(cpp-phmap INTERFACE IMPORTED GLOBAL)
  target_include_directories(cpp-phmap SYSTEM INTERFACE ${cpp-phmap_SOURCE_DIR})
  add_library(phpmap::phpmap ALIAS cpp-phmap)
endif()

mark_as_advanced(
  FETCHCONTENT_SOURCE_DIR_CPP-PHMAP
  FETCHCONTENT_UPDATES_DISCONNECTED_CPP-PHMAP
)

FetchContent_Declare(
  cpp-btree
  GIT_REPOSITORY https://github.com/ahueck/cpp-btree.git
)

FetchContent_GetProperties(cpp-btree)

if(NOT cpp-btree_POPULATED)
  FetchContent_Populate(cpp-btree)

  add_library(cppbtree INTERFACE IMPORTED GLOBAL)
  target_include_directories(cppbtree SYSTEM INTERFACE ${cpp-btree_SOURCE_DIR})
  #target_compile_features(cppbtree INTERFACE cxx_std_11 )
  add_library(google::btree ALIAS cppbtree)
endif()

mark_as_advanced(
  FETCHCONTENT_SOURCE_DIR_CPP-BTREE
  FETCHCONTENT_UPDATES_DISCONNECTED_CPP-BTREE
)
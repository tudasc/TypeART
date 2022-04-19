FetchContent_Declare(
  phmap_phmap
  GIT_REPOSITORY https://github.com/greg7mdp/parallel-hashmap.git
  GIT_SHALLOW 1
  GIT_TAG 48e86db1f4ebc85145d384ba061c4f615cb3106f
)

FetchContent_GetProperties(phmap_phmap)

if(NOT phmap_phmap_POPULATED)
  FetchContent_Populate(phmap_phmap)

  add_library(phmap_phmap INTERFACE IMPORTED GLOBAL)
  target_include_directories(phmap_phmap SYSTEM INTERFACE ${phmap_phmap_SOURCE_DIR})
  add_library(phpmap::phpmap ALIAS phmap_phmap)
endif()

mark_as_advanced(
  FETCHCONTENT_SOURCE_DIR_PHMAP_PHMAP
  FETCHCONTENT_UPDATES_DISCONNECTED_PHMAP_PHMAP
)

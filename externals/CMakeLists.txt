set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE STRING "" FORCE)
include(FetchContent)

mark_as_advanced(
  FETCHCONTENT_BASE_DIR
  FETCHCONTENT_QUIET
  FETCHCONTENT_FULLY_DISCONNECTED
  FETCHCONTENT_UPDATES_DISCONNECTED
)

if(TYPEART_BTREE)
  add_subdirectory(btree)
endif()

if(TYPEART_ABSEIL)
  add_subdirectory(abseil)
endif()

if (TYPEART_SAFEPTR)
  add_subdirectory(safe_ptr)
endif()

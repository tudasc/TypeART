set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE STRING "" FORCE)
include(FetchContent)

mark_as_advanced(
  FETCHCONTENT_BASE_DIR
  FETCHCONTENT_QUIET
  FETCHCONTENT_FULLY_DISCONNECTED
  FETCHCONTENT_UPDATES_DISCONNECTED
)

if(USE_BTREE)
  add_subdirectory(btree)
endif()

if(USE_ABSL)
  add_subdirectory(abseil)
endif()

if (ENABLE_SAFEPTR)
  add_subdirectory(safe_ptr)
endif()

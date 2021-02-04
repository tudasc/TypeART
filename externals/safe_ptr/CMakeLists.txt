FetchContent_Declare(
        safe_ptr
        GIT_REPOSITORY https://github.com/AlexeyAB/object_threadsafe.git
)

FetchContent_GetProperties(safe_ptr)
if(NOT safe_ptr_POPULATED)
    FetchContent_Populate(safe_ptr)
    file(COPY ${safe_ptr_SOURCE_DIR}/safe_ptr.h DESTINATION ${safe_ptr_BINARY_DIR})
    set(safe_ptr_include_dir ${safe_ptr_BINARY_DIR})
endif()

mark_as_advanced(
        FETCHCONTENT_SOURCE_DIR_SAFE_PTR
        FETCHCONTENT_UPDATES_DISCONNECTED_SAFE_PTR
)
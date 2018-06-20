set(RUNTIME_LIB_SOURCES
  Runtime.cpp
)

add_library(typeart SHARED
  ${RUNTIME_LIB_SOURCES}
)

target_link_libraries(typeart typelib)

target_include_directories(typeart PUBLIC
  ${CMAKE_SOURCE_DIR}/typelib
)

target_define_file_basename(typeart)

target_link_libraries(typeart
  LLVMCore
  LLVMSupport
)

make_tidy_check(typeart
  "${RUNTIME_LIB_SOURCES}"
)
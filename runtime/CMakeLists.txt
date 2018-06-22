set(RUNTIME_LIB_SOURCES
  Runtime.cpp
)

add_library(typeart SHARED
  ${RUNTIME_LIB_SOURCES}
)

target_link_libraries(typeart typelib)

target_include_directories(typeart 
  PUBLIC
    ${CMAKE_SOURCE_DIR}/typelib
  PRIVATE
    ${CMAKE_SOURCE_DIR}/lib # For the logger
)

target_include_directories(typeart 
  SYSTEM 
  PRIVATE
    ${LLVM_INCLUDE_DIRS}
)

target_define_file_basename(typeart)

target_link_libraries(typeart
  LLVMCore
  LLVMSupport
)

make_tidy_check(typeart
  "${RUNTIME_LIB_SOURCES}"
)
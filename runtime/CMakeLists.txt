set(RUNTIME_LIB_SOURCES
  Runtime.cpp
)

add_library(mustsupport SHARED 
  ${RUNTIME_LIB_SOURCES}
)

target_link_libraries(mustsupport musttypelib)

target_include_directories(mustsupport PUBLIC 
  ${CMAKE_SOURCE_DIR}/typelib
)
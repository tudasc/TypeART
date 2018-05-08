add_library(mustsupport SHARED Runtime.cpp)

target_link_libraries(mustsupport musttypelib)
target_include_directories(mustsupport PUBLIC "${CMAKE_SOURCE_DIR}/typelib")
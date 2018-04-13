add_library(mustsupport SHARED runtime.cpp)

target_link_libraries(mustsupport mustconfigio)
target_include_directories(mustsupport PUBLIC "${CMAKE_SOURCE_DIR}/configio")
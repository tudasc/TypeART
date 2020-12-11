add_subdirectory(passes)
add_subdirectory(typelib)
add_subdirectory(runtime)
add_subdirectory(mpi_interceptor)

add_format_target(format-sources
  "Formats project source files"
  TARGETS
  passes/*.cpp
  passes/*.h
  runtime/*.cpp
  runtime/*.h
  typelib/*.cpp
  typelib/*.h
  support/*.cpp
  support/*.h
)
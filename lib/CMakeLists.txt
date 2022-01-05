add_subdirectory(support)
add_subdirectory(passes)
add_subdirectory(typelib)
add_subdirectory(runtime)

if(${TYPEART_MPI_INTERCEPT_LIB})
  add_subdirectory(mpi_interceptor)
endif()

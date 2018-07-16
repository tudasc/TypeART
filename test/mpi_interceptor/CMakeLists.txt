find_package(MPI QUIET)
find_package(PythonInterp QUIET)

if(${MPI_C_FOUND} AND ${PYTHONINTERP_FOUND})
  set(LIB_SOURCE
    ${CMAKE_CURRENT_BINARY_DIR}/mpi_interceptor_rt.c
  )
  
  set(LIB_TMPL
    mpi_interceptor_tmpl.impl
  )

  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} wrap.py ${LIB_TMPL} -o ${LIB_SOURCE}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

  add_library(interceptor_rt
    ${LIB_SOURCE}
  )

  target_project_compile_options(interceptor_rt
    PRIVATE_DEFS MPI_LOGGER=1
  )

  target_define_file_basename(interceptor_rt)
  
  target_include_directories(interceptor_rt
    PUBLIC
      ${PROJECT_SOURCE_DIR}/runtime
      ${CMAKE_CURRENT_SOURCE_DIR}
  )
  
  target_include_directories(interceptor_rt
	SYSTEM 
	PRIVATE
	  ${MPI_C_INCLUDE_PATH}
	  ${LLVM_INCLUDE_DIRS}
  )

  target_link_libraries(interceptor_rt
    typeart-rt
    ${MPI_C_LIBRARIES}
  )
else()
  message(STATUS "Cannot build mpi interceptor_rt lib. MPI found: " ${MPI_C_FOUND} " and python found: " ${PYTHONINTERP_FOUND})
endif()
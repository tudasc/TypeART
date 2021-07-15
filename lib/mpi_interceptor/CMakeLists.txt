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

  add_library(PMPIToolRuntime SHARED
    ${LIB_SOURCE}
  )
  set_property(TARGET PMPIToolRuntime PROPERTY OUTPUT_NAME "${PROJECT_NAME}PMPIToolRuntime")
  add_library(typeart::PMPIToolRuntime ALIAS PMPIToolRuntime)


  set_property(TARGET PMPIToolRuntime PROPERTY C_STANDARD 11)
  set_property(TARGET PMPIToolRuntime PROPERTY C_STANDARD_REQUIRED TRUE)

  target_define_file_basename(PMPIToolRuntime)

  target_include_directories(PMPIToolRuntime
    PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  )

  target_include_directories(PMPIToolRuntime
    SYSTEM
    PRIVATE
      ${LLVM_INCLUDE_DIRS}
  )

  if(ENABLE_TSAN)
    target_tsan_options(PMPIToolRuntime)
  endif()

  target_link_libraries(PMPIToolRuntime
    PRIVATE
      typeart::Runtime
      MPI::MPI_C
  )

  install(
    TARGETS PMPIToolRuntime
    EXPORT ${TARGETS_EXPORT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
else()
  message(STATUS "Cannot build mpi PMPIToolRuntime lib. MPI found: " ${MPI_C_FOUND} " and python found: " ${PYTHONINTERP_FOUND})
endif()

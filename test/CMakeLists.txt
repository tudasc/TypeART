typeart_find_llvm_progs(TYPEART_FILECHECK_EXEC "FileCheck-${LLVM_VERSION_MAJOR};FileCheck" ABORT_IF_MISSING)

find_package(Python3 COMPONENTS Interpreter)

if(LLVM_EXTERNAL_LIT)
  cmake_path(GET LLVM_EXTERNAL_LIT PARENT_PATH LLVM_EXTERNAL_LIT_DIR)
endif()

typeart_find_llvm_progs(TYPEART_LIT_EXEC
  "llvm-lit;lit;lit.py"
  HINTS ${LLVM_EXTERNAL_LIT_DIR} /usr/lib/llvm-${LLVM_VERSION_MAJOR} /usr/lib/llvm /usr/bin /usr/local/bin /opt/local/bin
  ABORT_IF_MISSING
)

# add_subdirectory(compat)

function(cuda_runnable_detect cuda_is_runnable)
  set(${cuda_is_runnable} 0 PARENT_SCOPE)
  if(NOT CUDAToolkit_FOUND)
    return()
  endif()

  find_program(TYPEART_NVIDIA_SMI_EXEC nvidia-smi)
  mark_as_advanced(TYPEART_NVIDIA_SMI_EXEC)
  if(NOT TYPEART_NVIDIA_SMI_EXEC)
    return()
  endif()

  execute_process(
    COMMAND ${TYPEART_NVIDIA_SMI_EXEC} -L
    RESULT_VARIABLE smi_result
    OUTPUT_VARIABLE smi_out
    ERROR_QUIET
  )
  if(NOT smi_result EQUAL 0)
    return()
  endif()

  string(REGEX MATCHALL "GPU [0-9]+:" smi_gpu_list ${smi_out})
  list(LENGTH smi_gpu_list TYPEART_CUDA_GPU_COUNT)
  if(TYPEART_CUDA_GPU_COUNT GREATER 0)
    set(${cuda_is_runnable} 1 PARENT_SCOPE)
  endif()
endfunction()

cuda_runnable_detect(TYPEART_CUDA_RUNNABLE)

function(hip_runnable_detect hip_is_runnable)
  set(${hip_is_runnable} 0 PARENT_SCOPE)
  if(NOT hip_FOUND)
    return()
  endif()

  find_program(TYPEART_ROCM_SMI_EXEC rocm-smi)
  mark_as_advanced(TYPEART_ROCM_SMI_EXEC)
  if(NOT TYPEART_ROCM_SMI_EXEC)
    return()
  endif()

  execute_process(
    COMMAND ${TYPEART_ROCM_SMI_EXEC} -L
    RESULT_VARIABLE smi_result
    OUTPUT_VARIABLE smi_out
    ERROR_QUIET
  )
  if(NOT smi_result EQUAL 0)
    return()
  endif()

  string(REGEX MATCHALL "GPU [0-9]+:" smi_gpu_list ${smi_out})
  list(LENGTH smi_gpu_list TYPEART_HIP_GPU_COUNT)
  if(TYPEART_HIP_GPU_COUNT GREATER 0)
    set(${hip_is_runnable} 1 PARENT_SCOPE)
  endif()
endfunction()

hip_runnable_detect(TYPEART_HIP_RUNNABLE)

macro(pythonize_bool truth_var var)
  if(${truth_var})
    set(${var} True)
  else()
    set(${var} False)
  endif()
endmacro()

function(typeart_configure_lit_site input output)
  # This function provides an automatic way to 'configure'-like generate a file
  # based on a set of common and custom variables, specifically targeting the
  # variables needed for the 'lit.site.cfg' files. This function bundles the
  # common variables that any Lit instance is likely to need, and custom
  # variables can be passed in.

  set(TYPEARTPASS_PROJECT_DIR ${PROJECT_SOURCE_DIR})
  set(TYPEARTPASS_BINARY_DIR ${CMAKE_BINARY_DIR})
  set(TYPEARTPASS_SCRIPT_DIR ${CMAKE_BINARY_DIR}/scripts)
  set(TYPEARTPASS_LIBRARY_DIR ${CMAKE_BINARY_DIR}/lib/passes)
  set(TYPEARTPASS_BASE_LIBRARY_DIR ${CMAKE_BINARY_DIR}/lib)
  set(TYPEARTPASS_RT_DIR ${CMAKE_BINARY_DIR}/lib/runtime)
  set(TYPEARTPASS_PROFILE_FILE ${TYPEART_PROFILE_DIR}/lit-code-%p.profraw)

  pythonize_bool(TYPEART_SOFTCOUNTERS TYPEARTPASS_SOFTCOUNTER)

  pythonize_bool(OPENMP_FOUND TYPEARTPASS_OPENMP)
  pythonize_bool(Threads_FOUND TYPEARTPASS_THREADS)
  pythonize_bool(TYPEART_DISABLE_THREAD_SAFETY TYPEARTPASS_THREAD_UNSAFE)

  set(TYPEARTPASS_CUDA_FLAGS "")
  set(TYPEARTPASS_CUDA_PATH "")
  
  if(CUDAToolkit_FOUND)
    if(DEFINED CUDAToolkit_TARGET_DIR AND NOT "${CUDAToolkit_TARGET_DIR}" STREQUAL "")
      set(TYPEARTPASS_CUDA_PATH "${CUDAToolkit_TARGET_DIR}")
    elseif(DEFINED CUDAToolkit_LIBRARY_ROOT AND NOT "${CUDAToolkit_LIBRARY_ROOT}" STREQUAL "")
      set(TYPEARTPASS_CUDA_PATH "${CUDAToolkit_LIBRARY_ROOT}")
    elseif(DEFINED CUDAToolkit_BIN_DIR AND NOT "${CUDAToolkit_BIN_DIR}" STREQUAL "")
      cmake_path(GET CUDAToolkit_BIN_DIR PARENT_PATH TYPEARTPASS_CUDA_PATH)
    endif()

    foreach(cuda_inc ${CUDAToolkit_INCLUDE_DIRS})
      string(APPEND TYPEARTPASS_CUDA_FLAGS " -isystem ${cuda_inc}")
    endforeach()
  else()
    find_path(TYPEART_CUDA_RUNTIME_HEADER_DIR cuda_runtime_api.h
      HINTS /usr/local/cuda/include /usr/include /opt/cuda/include
    )
    if(TYPEART_CUDA_RUNTIME_HEADER_DIR)
      cmake_path(GET TYPEART_CUDA_RUNTIME_HEADER_DIR PARENT_PATH TYPEARTPASS_CUDA_PATH)
    endif()
  endif()

  set(TYPEART_CUDA_STATIC_AVAILABLE False)
  if(CUDAToolkit_FOUND OR NOT "${TYPEARTPASS_CUDA_PATH}" STREQUAL "")
    set(TYPEART_CUDA_STATIC_AVAILABLE True)
  endif()

  set(TYPEART_HIP_STATIC_AVAILABLE False)
  if(hip_FOUND)
    set(TYPEART_HIP_STATIC_AVAILABLE True)
  endif()

  pythonize_bool(TYPEART_CUDA_STATIC_AVAILABLE TYPEARTPASS_CUDA_STATIC)
  pythonize_bool(TYPEART_CUDA_RUNNABLE TYPEARTPASS_CUDA_RUNTIME)

  pythonize_bool(MPI_C_FOUND TYPEARTPASS_MPI_C)
  pythonize_bool(MPI_CXX_FOUND TYPEARTPASS_MPI_CXX)
  pythonize_bool(TYPEART_CI_RUN TYPEARTPASS_CI_RUN)

  pythonize_bool(TYPEART_TSAN TYPEARTPASS_TSAN)
  pythonize_bool(TYPEART_ASAN TYPEARTPASS_ASAN)
  pythonize_bool(TYPEART_UBSAN TYPEARTPASS_UBSAN)

  set(TYPEART_COVERAGE False)
  if(TYPEART_CODE_COVERAGE OR TYPEART_LLVM_CODE_COVERAGE)
    set(TYPEART_COVERAGE True)
  endif()

  set(TYPEART_USE_NEW_PM True)
  set(TYPEART_OPT_EXEC "${CMAKE_BINARY_DIR}/scripts/opt-shim")

  pythonize_bool(TYPEART_USE_LEGACY_WRAPPER TYPEARTPASS_LEGACY_WRAPPER)

  set(LIT_SITE_CFG_IN_HEADER
      "## Autogenerated by TypeART generation from ${input}\n## Do not edit!"
  )

  typeart_target_generate_file(${input} ${output})
endfunction()

typeart_add_format_target(typeart-format-tests
  "Formats project test source files"
  TARGETS *.c *.cpp *.h
)

typeart_configure_lit_site(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
)

function(typeart_add_lit_testsuite target comment)
  cmake_parse_arguments(ARG "" "" "PARAMS;DEPENDS;ARGS" ${ARGN})

  foreach(param ${ARG_PARAMS})
    list(APPEND TEST_PARAMS --param ${param})
  endforeach()

  add_custom_target(${target}
    COMMAND ${TYPEART_LIT_EXEC} ${ARG_ARGS} ${TEST_PARAMS} ${ARG_UNPARSED_ARGUMENTS}
    COMMENT "${comment}"
    USES_TERMINAL
  )

  if (ARG_DEPENDS)
    add_dependencies(${target} ${ARG_DEPENDS})
  endif()
endfunction()

function(typeart_add_lit_target)
  cmake_parse_arguments(ARG "" "" "SUITES;WORKERS" ${ARGN})

  foreach(suite workers IN ZIP_LISTS ARG_SUITES ARG_WORKERS)
    if("${suite}" STREQUAL "all")
      set(SUITE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
      set(TARGET_NAME check-typeart)
    else()
      if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${suite})
        continue()
      endif()
      set(SUITE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${suite})
      set(TARGET_NAME check-typeart-${suite})
    endif()

    if(NOT EXISTS ${SUITE_PATH})
      message(WARNING "Could not find suitable lit test target at ${SUITE_PATH}")
      continue()
    endif()

    typeart_add_lit_testsuite(${TARGET_NAME}
      "Running the lit suite TypeART::${suite}"
      ${SUITE_PATH}
      ARGS --show-unsupported -v -j ${workers}
      PARAMS typeartpass_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
      DEPENDS ${TYPEART_TEST_DEPENDS}
    )
  endforeach()
endfunction()

set(TYPEART_TEST_DEPENDS
  typeart::TransformPass
  typeart::Runtime
  typeart::Types
)

include(ProcessorCount)
ProcessorCount(NUM_CPU)
if(NUM_CPU EQUAL 0)
  set(NUM_CPU 1)
elseif(NUM_CPU GREATER_EQUAL 8)
  set(NUM_CPU 8)
endif()

set(TYPEART_SUITES
  all
  pass
  cuda
  hip
  runtime
  script
  typemapping
  runtime_inlined_types
  staging
)

set(TYPEART_SUITES_WORKERS
  1
  ${NUM_CPU}
  1
  1
  1
  1
  ${NUM_CPU}
  1
  1
)

typeart_add_lit_target(SUITES ${TYPEART_SUITES} WORKERS ${TYPEART_SUITES_WORKERS})

add_test(
  NAME typeart-lit-suite
  COMMAND
    ${TYPEART_LIT_EXEC} -j 1 --param
    typeartpass_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(TYPEART_TEST_CONFIGURE_IDE)
  function(add_test_target_ide target header sources compile_std linker_language)
    if(NOT sources)
      return()
    endif()

    add_executable(${target} EXCLUDE_FROM_ALL ${header} ${sources})

    target_include_directories(
      ${target} PRIVATE ${PROJECT_SOURCE_DIR}/lib/typelib
                        ${PROJECT_SOURCE_DIR}/lib/runtime
                        ${PROJECT_SOURCE_DIR}
    )
    target_compile_features(${target} PUBLIC ${compile_std})
    set_target_properties(${target} PROPERTIES LINKER_LANGUAGE ${linker_language})
  endfunction()

  file(GLOB_RECURSE TYPEART_CXX_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
  file(GLOB_RECURSE TYPEART_CC_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.c)
  file(GLOB_RECURSE TYPEART_HEADER_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

  add_test_target_ide(typeart_cxx_test_objects
    "${TYPEART_HEADER_TESTS}" "${TYPEART_CXX_TESTS}"
    cxx_std_17 CXX
  )
  add_test_target_ide(typeart_cc_test_objects
    "${TYPEART_HEADER_TESTS}" "${TYPEART_CC_TESTS}"
    c_std_99 C
  )
endif()

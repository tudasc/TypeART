cmake_minimum_required(VERSION 3.12)

project(typeart-lulesh-integration-tests)

set(WORK_DIR ${PROJECT_SOURCE_DIR})
set(TEST_SCRIPT ${WORK_DIR}/run_integration_test.sh)
set(LULESH_NP 8 CACHE STRING "Number of MPI processes for lulesh")
mark_as_advanced(LULESH_NP)

function(add_integration_test name dir exe_dir np args)
    message(STATUS "Configuring benchmark ${name}")

    add_custom_target(build_${name}
        COMMAND "make" TYPEART=yes
        WORKING_DIRECTORY ${dir}
    )
    
    set_target_properties(build_${name} PROPERTIES ENVIRONMENT TYPEART_PATH=${CMAKE_BINARY_DIR})

    
    add_custom_target(clean_${name}
        COMMAND "make" clean
        WORKING_DIRECTORY ${dir}
    )
    
    add_test(NAME test_build_${name}
        COMMAND "make" TYPEART=yes
        WORKING_DIRECTORY ${dir}
    )
        
    add_test(NAME test_clean_${name}
        COMMAND "make" clean
        WORKING_DIRECTORY ${dir}
    )
    
    set_tests_properties(test_build_${name} PROPERTIES ENVIRONMENT TYPEART_PATH=${CMAKE_BINARY_DIR})
    set_tests_properties(test_build_${name} PROPERTIES DEPENDS test_clean_${name})
    set_tests_properties(test_clean_${name} PROPERTIES FIXTURES_SETUP lulesh_b)
    set_tests_properties(test_build_${name} PROPERTIES FIXTURES_SETUP lulesh_b)
    #set_tests_properties(test_clean_${name} PROPERTIES FIXTURES_CLEANUP lulesh_b)
    
    add_test(NAME ${name}
        COMMAND "${TEST_SCRIPT}" ${name} ${np} ${args} ${CMAKE_BINARY_DIR}/lib/runtime ${CMAKE_BINARY_DIR}/lib/mpi_interceptor
        WORKING_DIRECTORY "${exe_dir}"
    )
    
    set_tests_properties(${name} PROPERTIES FIXTURES_REQUIRED lulesh_b)
    set_tests_properties(${name} PROPERTIES ENVIRONMENT LLVM_PROFILE_FILE=${TYPEART_PROFILE_DIR}/code-%p-lulesh.profraw)
endfunction()

if(TYPEART_TSAN)
    message(STATUS  "TSAN with lulesh is currently unsupported")
else()
    add_integration_test("lulesh2.0" "${WORK_DIR}" "${WORK_DIR}" ${LULESH_NP} "-s 12 -i 200")
endif()
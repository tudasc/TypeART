


add_executable(demo 01_struct_example.c)

target_link_libraries(demo typeart)

add_custom_target(run
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/01_ex)
cmake_minimum_required(VERSION 3.10)
project(mini-cpr VERSION 0.1 DESCRIPTION "A basic Checkpoint/Restart Library written in C" LANGUAGES C)

# test executable
add_executable(test-mini-cpr
        src/test/test-mini-cpr.c
        src/mini-cpr.c
        src/mini-cpr.h
        src/mini-cpr_p.h
        src/test/include/unity.h
        src/test/include/unity_internals.h
        src/test/include/unity.c)
target_compile_options(test-mini-cpr PRIVATE -O0 -g)

add_library(minicpr
	SHARED
	src/mini-cpr.c
	src/mini-cpr.h
	src/mini-cpr_p.h
	)

# test app
add_executable(test-app
        src/test/test-app.c
        src/mini-cpr.c
        src/mini-cpr.h
        src/mini-cpr_p.h
        src/test/include/unity.h
        src/test/include/unity_internals.h
        src/test/include/unity.c)
target_compile_options(test-app PRIVATE -O0 -g)

add_executable(test-app-shared
	src/test/test-app.c
	src/test/include/unity.c
	)

target_link_libraries(test-app-shared
	minicpr
	)

# copy test assets to build directory
file(COPY src/test/assets DESTINATION ${CMAKE_BINARY_DIR})

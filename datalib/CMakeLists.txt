set(PROJECT_NAME typeart_datalib)
set(TARGETS_EXPORT_NAME ${PROJECT_NAME}-targets)

set(DATA_LIB_SOURCES
        DataDB.cpp
        DataIO.cpp
        TaData.h)

add_library(datalib STATIC
        ${DATA_LIB_SOURCES}
        )

set_target_properties(datalib
        PROPERTIES POSITION_INDEPENDENT_CODE ON
        )

target_include_directories(datalib
        SYSTEM
        PRIVATE
        ${LLVM_INCLUDE_DIRS}
        )

target_project_compile_options(datalib)
target_project_compile_definitions(datalib
        ARG_PRIVATE_DEFS
        LOG_LEVEL=${LOG_LEVEL}
        )

make_tidy_check(datalib
        "${DATA_LIB_SOURCES}"
        )

install(
        TARGETS datalib
        EXPORT ${TARGETS_EXPORT_NAME}
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        INCLUDES DESTINATION
        include
)

install(
        FILES DataInterface.h DataIO.h DataDB.h
        DESTINATION include
)

install(
        EXPORT ${TARGETS_EXPORT_NAME}
        NAMESPACE typeart::
        DESTINATION lib/cmake
)

configure_package_config_file(
        ${PROJECT_SOURCE_DIR}/cmake/Config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        INSTALL_DESTINATION lib/cmake
)

install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        DESTINATION lib/cmake
        )
set(PROJECT_NAME tycart-rt)
set(TARGETS_EXPORT_NAME ${PROJECT_NAME}-targets)

set(TYCART_LIB_SOURCES
        TyCartRT.cpp
        )

add_library(tycart-rt SHARED
        ${TYCART_LIB_SOURCES}
        )

if(WITH_FTI)
    set(CPR_BACKEND_INCLUDE ${FTI_INSTALL_DIR}/include)
    set(CPR_BACKEND_LIBRARIES ${FTI_INSTALL_DIR}/lib/libfti.so)
    target_compile_definitions(tycart-rt PRIVATE WITH_FTI)
endif()

if(WITH_VELOC)
    set(CPR_BACKEND_INCLUDE ${VELOC_INSTALL_DIR}/include)
    set(CPR_BACKEND_LIBRARIES ${VELOC_INSTALL_DIR}/lib/libveloc-client.so
            ${VELOC_INSTALL_DIR}/lib/libveloc-modules.so)
    target_compile_definitions(tycart-rt PRIVATE WITH_VELOC)
endif()

if(WITH_MINI_CPR)
    set(CPR_BACKEND_INCLUDE ${MINI_INSTALL_DIR}/src)
    set(CPR_BACKEND_LIBRARIES ${MINI_INSTALL_DIR}/build/libminicpr.so)
    target_compile_definitions(tycart-rt PRIVATE WITH_MINI_CPR)
endif()

target_link_libraries(tycart-rt
        # Include VeloC / FTI
        ${CPR_BACKEND_LIBRARIES}
        )

target_include_directories(tycart-rt
        PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/runtime/tycart>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/runtime>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/typelib>
        $<INSTALL_INTERFACE:include>
        )

target_include_directories(tycart-rt
        SYSTEM
        PRIVATE
        ${LLVM_INCLUDE_DIRS}
        ${CPR_BACKEND_INCLUDE}
        )

target_project_compile_definitions(tycart-rt
        PRIVATE_DEFS
        LOG_LEVEL=${LOG_LEVEL_RT}
        )

target_project_compile_definitions(tycart-rt)
target_define_file_basename(tycart-rt)

make_tidy_check(tycart-rt
        "${TYCART_LIB_SOURCES}"
        )

install(
        TARGETS tycart-rt
        EXPORT ${TARGETS_EXPORT_NAME}
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static
        #  INCLUDES DESTINATION
        #    include # --> superfluous since $<INSTALL_INTERFACE:include> is used
)

install(
        FILES tycart.h typecheck_macro.h
        DESTINATION include
)

install(
        EXPORT ${TARGETS_EXPORT_NAME}
        NAMESPACE tycart::
        DESTINATION lib/cmake
)

#set(DEPENDENCY libaTargets.cmake
configure_package_config_file(
        ${PROJECT_SOURCE_DIR}/cmake/Config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        INSTALL_DESTINATION lib/cmake
)

install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        DESTINATION lib/cmake
        )
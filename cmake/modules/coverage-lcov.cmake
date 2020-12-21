find_program(LCOV_COMMAND lcov)
find_program(GENHTML_COMMAND genhtml)

if(LCOV_COMMAND-NOTFOUND or GENHTML_COMMAND-NOTFOUND)
  message(WARNING "lcov and genhtml command needed for coverage.")
endif()

add_custom_target(
  lcov-clean
  COMMAND ${LCOV_COMMAND} -d ${CMAKE_BINARY_DIR} -z
)

add_custom_target(
  lcov-make
  COMMAND ${LCOV_COMMAND} --no-external -c -d ${CMAKE_BINARY_DIR} -b ${CMAKE_SOURCE_DIR} -o counter.pro
  COMMAND ${LCOV_COMMAND} --remove counter.pro '${CMAKE_BINARY_DIR}/*' -o counter.pro
)

add_custom_target(
  lcov-html
  COMMAND ${GENHTML_COMMAND} -o ${TYPEART_PROFILE_DIR} counter.pro
)


function(make_lcov_target target)
#  add_custom_target(
#    lcov-clean-${target}
#    COMMAND lcov -d ${CMAKE_BINARY_DIR} -z
#    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
#  )

  get_target_property(LCOV_TARGET_SOURCE_DIR ${target} SOURCE_DIR)

  add_custom_target(
    lcov-make-${target}
    COMMAND ${LCOV_COMMAND} --no-external -c -d ${CMAKE_BINARY_DIR} -b ${LCOV_TARGET_SOURCE_DIR} -o counter-${target}.pro
    COMMAND ${LCOV_COMMAND} --remove counter-${target}.pro '${CMAKE_BINARY_DIR}/*' -o counter-${target}.pro
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  add_custom_target(
    lcov-html-${target}
    COMMAND ${GENHTML_COMMAND} -o ${TYPEART_PROFILE_DIR} counter-${target}.pro
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS lcov-make-${target}
  )
endfunction()

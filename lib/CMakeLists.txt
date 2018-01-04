list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(AddLLVM)

set(PASS_SOURCES
 	MUSTSupportPass.cpp
	support/TypeUtil.cpp
	MemOpVisitor.cpp
)

add_llvm_loadable_module(MustSupportPass
	${PASS_SOURCES}
)

add_tidy_target(tidy-run-on-pass
  "Clang-tidy run on MUST support pass translation units"
  SOURCES ${PASS_SOURCES}
  OTHER --header-filter=${CMAKE_CURRENT_SOURCE_DIR}
)

add_tidy_fix_target(tidy-fix-on-pass
  "Clang-tidy run with fixes on MUST support pass translation units"
  SOURCES ${PASS_SOURCES}
  OTHER --header-filter=${CMAKE_CURRENT_SOURCE_DIR} -checks=-*,modernize-*,llvm-namespace-comment,google-explicit-constructor
)

set(MUSTPASS_TEST_DEPS MustSupportPass)
add_lit_testsuite(lit-pass-test
  "Running the MUST support pass tests"
  ${PROJECT_SOURCE_DIR}/test
  PARAMS ${MUSTPASS_TEST_PARAMS}
  DEPENDS ${MUSTPASS_TEST_DEPS}
  ARGS ${MUSTPASS_TEST_EXTRA_ARGS}
)
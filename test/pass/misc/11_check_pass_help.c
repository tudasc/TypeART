// RUN: %c-to-llvm %s | %apply-typeart -help-hidden -S 2>&1 | %filecheck %s

void foo() {
}

// clang-format off

// CHECK:   --typeart-config=<string>                                             - Location of the configuration file to configure the TypeART pass. Commandline arguments are prioritized.
// CHECK:   --typeart-config-dump                                                 - Dump default config file content to std::out.
// CHECK:   --typeart-global                                                      - Instrument global allocations.
// CHECK:   --typeart-heap                                                        - Instrument heap allocation/free instructions.
// CHECK:   --typeart-stack                                                       - Instrument stack allocations.
// CHECK:   --typeart-stack-lifetime                                              - Instrument lifetime.start intrinsic instead of alloca.
// CHECK:   --typeart-stats                                                       - Show statistics for TypeArt type pass.
// CHECK:   --typeart-types=<string>                                              - Location of the generated type file.
// CHECK: TypeART memory instruction finder:
// CHECK: These options control which memory instructions are collected/filtered.
// CHECK:   --typeart-analysis-filter-global                                      - Filter globals of a module.
// CHECK:   --typeart-analysis-filter-heap-alloca                                 - Filter alloca instructions that have a store from a heap allocation.
// CHECK:   --typeart-analysis-filter-non-array-alloca                            - Filter scalar valued allocas.
// CHECK:   --typeart-analysis-filter-pointer-alloca                              - Filter allocas of pointer types.
// CHECK:   --typeart-filter                                                      - Filter allocas (stack/global) that are passed to relevant function calls.
// CHECK:   --typeart-filter-cg-file=<string>                                     - Location of call-graph file to use.
// CHECK:   --typeart-filter-glob=<string>                                        - Filter allocas based on the function name (glob) <string>.
// CHECK:   --typeart-filter-glob-deep=<string>                                   - Filter allocas based on specific API: Values passed as void* are correlated when string matched and kept when correlated successfully.
// CHECK:   --typeart-filter-implementation=<value>                               - Select the call filter implementation.
// CHECK:     =none                                                               -   No filter
// CHECK:     =std                                                                -   Standard forward filter (default)
// CHECK:     =cg                                                                 -   Call-graph-based filter

// clang-format on
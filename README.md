# TypeART &middot; [![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) ![](https://github.com/tudasc/TypeART/actions/workflows/basic-ci.yml/badge.svg?branch=master) ![](https://github.com/tudasc/TypeART/actions/workflows/ext-ci.yml/badge.svg?branch=master) [![Coverage Status](https://coveralls.io/repos/github/tudasc/TypeART/badge.svg?branch=master)](https://coveralls.io/github/tudasc/TypeART?branch=master)

## What is TypeART?

TypeART \[[TA18](#ref-typeart-2018); [TA20](#ref-typeart-2020); [TA22](#ref-typeart-2022)\] is a type and memory
allocation tracking sanitizer based on the [LLVM](https://llvm.org) compiler toolchain for C/C++ (OpenMP) codes. It includes an LLVM compiler pass plugin for instrumentation and a runtime library to monitor memory allocations during program execution.

TypeART instruments heap, stack, and global variable allocations with callbacks to its runtime, capturing:
(1) the memory address, (2) the type-layout information of the allocation (e.g., built-ins, user-defined structs) and (3) number of elements.

## Why use it?

TypeART provides type-related information of allocations in your program to help verify some property, and to help
generate diagnostics if it doesn't hold.

Low-level C APIs often rely on `void*` pointers for generic types, requiring users to manually specify type and size - a process prone to errors. Examples for type unsafe APIs include the Message-Passing Interface (MPI),
checkpointing libraries and numeric solver libraries. 
TypeART simplifies verification, ensuring, for example, that a `void*` argument corresponds to an array of expected type `T` with length `n`.

### Use Case: MUST - A dynamic MPI correctness checker

MUST \[[MU13](#ref-must-2013)\], a dynamic MPI correctness checker, detects issues like deadlocks or mismatched MPI datatypes. For more details, visit its [project page](https://www.hpc.rwth-aachen.de/must/).

MUST intercepts MPI calls for analysis but cannot deduce the *effective* type of `void*` buffers in MPI APIs. TypeART addresses this by tracking memory (de-)allocations relevant to MPI communication in user code, allowing MUST to validate type compatibility between MPI buffers and declared datatypes.

#### Type checking for MPI calls

To demonstrate the utility of TypeART, consider the following code:

```c
// Otherwise unknown to MUST, TypeART tracks this allocation (memory address, type and size):
double* array = (double*) malloc(length*sizeof(double));
// MUST intercepts this MPI call, asking TypeART's runtime for type information:
//   1. Is the first argument of type double (due to MPI_DOUBLE)?
//   2. Is the allocation at least of size *length*? 
MPI_Send((void*) array, length, MPI_DOUBLE, ...)
```

MUST and TypeART also support MPI [derived datatypes](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node96.htm)
with complex underlying data structures. For further details, see
our [publications](#references), or download MUST (v1.8 or higher integrates TypeART) from
its [project page](https://itc.rwth-aachen.de/must/).

## Table of Contents

* [1. Using TypeART](#1-using-typeart)
    * [1.1 Compiling a target code](#11-compiling-a-target-code)
        * [1.1.1 Building with TypeART](#111-building-with-typeart)
        * [1.1.2 Options for TypeART passes and compiler wrapper](#112-options-for-typeart-passes-and-compiler-wrapper)
        * [1.1.3 Serialized type information](#113-serialized-type-information)
        * [1.1.4 Filtering allocations](#114-filtering-allocations)
    * [1.2 Executing an instrumented target code](#12-executing-an-instrumented-target-code)
    * [1.3 Example: MPI demo](#13-example-mpi-demo)
* [2. Building TypeART](#2-building-typeart)
    * [2.1 Optional software requirements](#21-optional-software-requirements)
    * [2.2 Building](#22-building)
        * [2.2.1 CMake configuration: Options for users](#221-cmake-configuration-options-for-users)
* [3. Consuming TypeART](#3-consuming-typeart)
* [References](#references)

## 1. Using TypeART

Using TypeART involves two phases:

1. Compilation: Compile your code with Clang/LLVM (version 14) using the TypeART LLVM pass plugin. The plugin (1) serializes static type information to a file and (2) instruments relevant allocations. See [Section 1.1](#11-compiling-a-target-code).
2. Execution: Run the instrumented program with a TypeART runtime client, which uses the callback data to perform analysis facilitating the static type information. See [Section 1.2](#12-executing-an-instrumented-target-code).

### 1.1 Compiling a target code

TypeART’s LLVM compiler pass plugins instrument allocations and serialize static type layouts into a YAML file (default: `types.yaml`). We provide compiler wrapper scripts (available in the bin folder of the TypeART installation) for Clang and MPI. By default, these wrappers instrument heap, stack, and global allocations, while MPI wrappers filter allocations unrelated to MPI calls (see [Section 1.1.4](#114-filtering-allocations)).

*Note*: Currently, the compilation must be serialized, e.g., `make -j 1`, to ensure consistent type information across translation units.

#### 1.1.1 Building with TypeART

A typical compile invocation may first compile code to object files and then link with any libraries:

```shell
# Compile:
$> clang++ -O2 $(COMPILE_FLAGS) -c code.cpp -o code.o
# Link:
$> clang++ $(LINK_FLAGS) code.o -o binary
```

With TypeART, the recipe needs to be changed to, e.g., use our provided compiler wrapper, as we rely on the LLVM `opt`
(optimizer) tool to load and apply our TypeART passes to a target code:

```shell
# Compile, replace direct clang++ call with wrapper of the TypeART installation:
$> typeart-clang++ -O2 $(COMPILE_FLAGS) -c code.cpp -o code.o
# Link, also with the wrapper:
$> typeart-clang++ $(LINK_FLAGS) code.o -o binary
```

The wrapper performs the following steps:

1. Compiles the code to LLVM IR and pipes it to the `opt` tool, retaining original compile flags.
2. Applies heap instrumentation with TypeART.
3. Optimizes the code using provided -O flag.
4. Applies stack and global instrumentation with TypeART.
5. Converts the LLVM IR to an object file using `llc`.
6. Links the TypeART runtime library with the provided linker flags.

*Note*: Heap allocations are instrumented before optimizations to prevent loss of type information in some cases.

##### Wrapper usage in CMake build systems

For plain Makefiles, the wrapper replaces the GCC/Clang compiler variables, e.g., `CC` or `MPICC`. For CMake, during the
configuration, it is advised to disable the wrapper temporarily. This is due to CMake executing internal compiler
checks, where we do not need TypeART instrumentation:

```shell
# Temporarily disable wrapper with environment flag TYPEART_WRAPPER=OFF for configuration:
$> TYPEART_WRAPPER=OFF cmake -B build -DCMAKE_C_COMPILER=*TypeART bin*/typeart-clang 
# Compile with typeart-clang:
$> cmake --build build --target install -- -j1
```

##### MPI wrapper generation

The wrappers `typeart-mpicc` and `typeart-mpic++` are generated for compiling MPI codes with TypeART.
Here, we rely on detecting the vendor to generate wrappers with appropriate environment variables to force the use of
the Clang/LLVM compiler.
We support detection for OpenMPI, Intel MPI and MPICH based on `mpi.h` symbols, and use the following flags for setting
the Clang
compiler:

| Vendor    | Symbol        | C compiler env. var  | C++ compiler env. var  |
|-----------|---------------|----------------------|------------------------|
| Open MPI  | OPEN_MPI      | OMPI_CC              | OMPI_CXX               |
| Intel MPI | I_MPI_VERSION | I_MPI_CC             | I_MPI_CXX              |
| MPICH     | MPICH_NAME    | MPICH_CC             | MPICH_CXX              |


#### 1.1.2 Options for TypeART passes and compiler wrapper

The pass behavior can be configured with the command line options listed below. TypeART also supports a configuration
file based on a yaml format. *Note:* For now, the TypeART pass prioritizes (1) commandline arguments and then (2)
environment files (if set) over the file-based configuration option.

##### Pass

For modification of the pass behavior, we provide several options. Some options have equivalent environment variables.
<!--- @formatter:off --->

| Flag                                         | Env. variable       |   Default    | Description                                                                                                                                        |
|----------------------------------------------|:--------------------|:------------:|----------------------------------------------------------------------------------------------------------------------------------------------------|
| `-typeart`                                   | -                   |      -       | Invoke TypeART pass through LLVM `opt`                                                                                                             |
| `--typeart-config`                           | `TYPEART_CONFIG`    |      -       | Pass configuration file (defines has all options below)                                                                                            |
| `--typeart-types`                            | `TYPEART_TYPE_FILE` | `types.yaml` | Serialized type layout information of user-defined types. File location and name can also be controlled with the env variable `TYPEART_TYPE_FILE`. |
| `--typeart-heap`                             | -                   |    `true`    | Instrument heap allocations                                                                                                                        |
| `--typeart-stack`                            | -                   |   `false`    | Instrument stack and global allocations. Enables instrumentation of global allocations.                                                            |
| `--typeart-stack-lifetime`                   | -                   |   `true`    | Instrument stack `llvm.lifetime.start` instead of `alloca` directly
| `--typeart-global`                           | -                   |   `false`    | Instrument global allocations (see --typeart-stack).                                                                                               |
| `--typeart-typegen`                          | -                   |   `dimeta`   | Values: `dimeta`, `ir`. The env variable `TYPEART_TYPEGEN_IR` set to 1 toggles `--typeart-typegen=ir`, i.e., serializing type information based on LLVM IR.  See [Section 1.1.3](#113-serialized-type-information). |
| `--typeart-stats`                            | -                   |   `false`    | Show instrumentation statistic counters                                                                                                            |
| `--typeart-filter`                           | -                   |   `false`    | Filter stack and global allocations. See also [Section 1.1.4](#114-filtering-allocations)                                                          |
| `--typeart-filter-implementation`            | -                   |    `std`     | Values: `std`, `none`. See also [Section 1.1.4](#114-filtering-allocations)                                                                        |
| `--typeart-filter-glob`                      | -                   |   `*MPI_*`   | Filter API string target (glob string)                                                                                                             |
| `--typeart-analysis-filter-global`           | -                   |    `true`    | Filter global alloca based on heuristics                                                                                                           |
| `--typeart-analysis-filter-heap-alloca`      | -                   |    `true`    | Filter stack alloca that have a store instruction from a heap allocation                                                                           |
| `--typeart-analysis-filter-non-array-alloca` | -                   |   `false`    | Filter scalar valued allocas                                                                                                                       |
| `--typeart-analysis-filter-pointer-alloca`   | -                   |    `true`    | Filter allocas of pointer types                                                                                                                    |

<!--- @formatter:on --->

##### Compiler wrapper

For modification of the pass behavior, the wrapper accepts configuration file commandline options. Equivalent
environment variables can be set.
<!--- @formatter:off --->

| Flag                            | Env. variable                   | Description                                                           |
|---------------------------------|---------------------------------|-----------------------------------------------------------------------|
| `--typeart-config=<file>`       | `TYPEART_WRAPPER_CONFIG`        | Pass yaml file configuration to heap and stack phase of TypeART pass. |
| `--typeart-heap-config=<file>`  | `TYPEART_WRAPPER_HEAP_CONFIG`   | See above, heap phase only.                                           |
| `--typeart-stack-config=<file>` | `TYPEART_WRAPPER_STACK_CONFIG`  | See above, stack/global phase only.                                   |

<!--- @formatter:on --->

##### Configuration file

The default file content of the configuration file is listed below. The option names correlate with the command line
options. Notably, e.g., the option `call-filter: { implementation: false }` correlates
to `--typeart-filter-implementation=false` etc.

```yaml
---
types: types.yaml
heap: true
stack: false
global: false
stats: false
stack-lifetime: true
typegen: dimeta
filter: true
call-filter:
  implementation: std
  glob: '*MPI_*'
  glob-deep: 'MPI_*'
  cg-file: ''
analysis:
  filter-global: true
  filter-heap-alloca: false
  filter-pointer-alloca: true
  filter-non-array-alloca: false
...

```

#### 1.1.3 Serialized type information

After instrumentation, the file `types.yaml` (`env TYPEART_TYPE_FILE`) contains the static type information. Each user-defined type layout is
extracted and an integer `type-id` is attached to it. Built-in types (e.g., float) have pre-defined ids and byte
layouts.
To generate these type layouts, TypeART is using either the [LLVM IR type system](https://llvm.org/docs/LangRef.html#type-system) (`--typeart-typegen=ir`), or using the external library [llvm-dimeta](https://github.com/ahueck/llvm-dimeta) (`--typeart-typegen=dimeta`) which extracts type information using [LLVM debug metadata](https://llvm.org/docs/SourceLevelDebugging.html).
The latter is default.

The TypeART instrumentation callbacks use the `type-id`. The runtime library correlates the allocation with the
respective type (and layout) during execution. Consider the following struct:

After instrumentation, the `types.yaml` file (also controlled via the `TYPEART_TYPE_FILE` environment variable) stores static type information. 
Each user-defined type layout is assigned a unique integer type-id. Built-in types (e.g., float) use predefined type-ids and byte layouts.

Type layouts are generated using one of the following methods:

* LLVM IR Type System: Extracts types directly from LLVM IR (`--typeart-typegen=ir`).
* LLVM Debug Metadata (Default): Extracts types using the llvm-dimeta library and LLVM debug metadata (`--typeart-typegen=dimeta`).

During execution, TypeART’s runtime library uses the type-id from callbacks to associate allocations with their type and layout. For example, consider the following struct:

```c
struct s1_t {
  char a[3];
  struct s1_t* b;
}
```

The TypeART pass may write a `types.yaml` file with the following content:
<!--- @formatter:off --->

```yaml
- id: 256            // struct type-id
  name: s1_t
  extent: 16         // size in bytes
  member_count: 2
  offsets: [ 0, 8 ]  // byte offsets from struct start
  types: [ 0, 10 ]   // member type-ids (0->char, 10->pointer)
  sizes: [ 3, 1 ]    // member (array) length
```

<!--- @formatter:on --->

##### Limitations

The type-id system is tailored for LLVM IR types, which imposes certain constraints. For instance, C/C++ types like unsigned integers are currently unsupported (and represented like signed integers). The list of supported built-in type-ids is defined in [TypeInterface.h](lib/typelib/TypeInterface.h) and reflects the types that TypeART can represent.


#### 1.1.4 Filtering allocations

To improve performance, a translation unit-local (TU) data-flow filter for global and stack variables exist. It follows
the LLVM IR `use-def` chain. If the allocation provably never reaches the target API, it can be filtered. Otherwise, it
is instrumented. Use the option `--typeart-filter` to filter and `--typeart-glob=<target API glob>` (default: `*MPI_*`)
to target the correct API.

Consider the following example.

```c
extern foo_bar(float*); // No definition in the TU 
void bar(float* x, float* y) {
  *x = 2.f; // x is not used after
  MPI_Send(y, ...);
}
void foo() {
  float a = 1.f, b = 2.f, c = 3.f;
  bar(&a, &b);
  foo_bar(&c);
}
```

1. The filter can remove `a`, as the aliasing pointer `x` is never part of an MPI call.
2. `b` is instrumented as the aliasing pointer `y` is part of an MPI call.
3. `c` is instrumented as we cannot reason about the body of `foo_bar`.

### 1.2 Executing an instrumented target code

To execute the instrumented code, the TypeART runtime library (or a derivative) has to be loaded to accept the
callbacks. The library also requires access to the `types.yaml` file to correlate the `type-id` with the actual type
layouts. To specify its path, you can use the environment variable `TYPEART_TYPE_FILE`, e.g.:

```shell
$> export TYPEART_TYPE_FILE=/path/to/types.yaml
# If the TypeART runtime is not resolved, LD_LIBRARY_PATH is set:
$> env LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(TYPEART_LIBPATH) ./binary
```

An example for pre-loading a TypeART-based library in the context of MPI is found in the demo,
see [Section 1.3](#13-example-mpi-demo).

### 1.3 Example: MPI demo

The folder [demo](demo) contains an example of MPI-related type errors that can be detected using TypeART. The code is
compiled with our instrumentation, and executed by preloading the MPI-related check library implemented
in [tool.c](demo/tool.c). The check library uses the TypeART [runtime query interface](lib/runtime/RuntimeInterface.h).
It overloads the required MPI calls and checks that the passed `void*` buffer is correct w.r.t. the MPI derived
datatype.

To compile and run the demo targets:

- Makefile
    ```shell
    # Valid MPI demo:
    $> MPICC=*TypeART prefix*/bin/typeart-mpicc make run-demo
    # Type-error MPI demo:
    $> MPICC=*TypeART prefix*/bin/typeart-mpicc make run-demo_broken
    ```
- CMake, likewise:
    ```shell
    $> TYPEART_WRAPPER=OFF cmake -S demo -B build_demo -DCMAKE_C_COMPILER=*TypeART prefix*/bin/typeart-mpicc 
    $> cmake --build build_demo --target run-demo
    $> cmake --build build_demo --target run-demo_broken
    ```

## 2. Building TypeART

TypeART supports LLVM version 14 (lower LLVM version support is deprecated) and CMake version >= 3.20.

### 2.1 Optional software requirements

- MPI library: (soft requirement) Needed for the MPI compiler wrappers, tests, the [demo](demo),
  our [MPI interceptor library](lib/mpi_interceptor), and for logging with our TypeART runtime library within an MPI
  target application.
- OpenMP-enabled Clang compiler: Needed for some tests.

Other smaller, external dependencies are defined within the [externals folder](externals) (depending on configuration
options), see [Section 2.2.1 (Runtime)](#221-cmake-configuration-options-for-users). They are automatically downloaded
during configuration time (internet connection required).

### 2.2 Building

TypeART uses CMake to build, cf. [GitHub CI build file](.github/workflows/basic-ci.yml) for a complete recipe to build.
Example build recipe (debug build, installs to default prefix
`${typeart_SOURCE_DIR}/install/typeart`)

```sh
$> git clone https://github.com/tudasc/TypeART
$> cd TypeART
$> cmake -B build
$> cmake --build build --target install --parallel
```

#### 2.2.1 CMake configuration: Options for users

##### Binaries (scripts)

<!--- @formatter:off --->

| Option                |  Default   | Description                                               |
|-----------------------|:----------:|-----------------------------------------------------------|
| `TYPEART_MPI_WRAPPER` |    `ON`    | Install TypeART MPI wrapper (mpic, mpic++). Requires MPI. |

<!--- @formatter:on --->

##### Runtime

<!--- @formatter:off --->

| Option                 | Default | Description                                                                                                             |
|------------------------|:-------:|-------------------------------------------------------------------------------------------------------------------------|
| `TYPEART_ABSEIL`       |  `ON`   | Enable usage of btree-backed map of the [Abseil project](https://abseil.io/) (LTS release) for storing allocation data. |
| `TYPEART_PHMAP`        |  `OFF`  | Enable usage of a [btree-backed map](https://github.com/greg7mdp/parallel-hashmap) (alternative to Abseil).             |
| `TYPEART_SOFTCOUNTERS` |  `OFF`  | Enable runtime tracking of #tracked addrs. / #distinct checks / etc.                                                    |
| `TYPEART_LOG_LEVEL_RT` |   `0`   | Granularity of runtime logger. 3 is most verbose, 0 is least.                                                           |

<!--- @formatter:on --->

###### Runtime thread-safety options

Default mode is to protect the global data structure with a (shared) mutex. Two main options exist:

<!--- @formatter:off --->

| Option                           | Default  | Description                                                                                                                                        |
|----------------------------------|:--------:|----------------------------------------------------------------------------------------------------------------------------------------------------|
| `TYPEART_DISABLE_THREAD_SAFETY`  |  `OFF`   | Disable thread safety of runtime                                                                                                                   |
| `TYPEART_SAFEPTR`                |  `OFF`   | Instead of a mutex, use a special data structure wrapper for concurrency, see [object_threadsafe](https://github.com/AlexeyAB/object_threadsafe)   |

<!--- @formatter:on --->

##### LLVM passes

<!--- @formatter:off --->

| Option                       | Default | Description                                                                                       |
|------------------------------|:-------:|---------------------------------------------------------------------------------------------------|
| `TYPEART_SHOW_STATS`         |  `ON`   | Passes show compile-time summary w.r.t. allocations counts                                        |
| `TYPEART_MPI_INTERCEPT_LIB`  |  `ON`   | Library to intercept MPI calls by preloading and check whether TypeART tracks the buffer pointer  |
| `TYPEART_MPI_LOGGER`         |  `ON`   | Enable better logging support in MPI execution context                                            |
| `TYPEART_LOG_LEVEL`          |   `0`   | Granularity of pass logger. 3 is most verbose, 0 is least                                         |

<!--- @formatter:on --->

##### Testing

<!--- @formatter:off --->

| Option                        | Default | Description                                                                                                  |
|-------------------------------|:-------:|--------------------------------------------------------------------------------------------------------------|
| `TYPEART_TEST_CONFIG`         |  `OFF`  | Enable testing, and set (force) logging levels to appropriate levels for test runner to succeed              |
| `TYPEART_CODE_COVERAGE`       |  `OFF`  | Enable code coverage statistics using LCOV 1.14 and genhtml (gcovr optional)                                 |
| `TYPEART_LLVM_CODE_COVERAGE`  |  `OFF`  | Enable llvm-cov code coverage statistics (llvm-cov and llvm-profdata  required)                              |
| `TYPEART_ASAN, TSAN, UBSAN`   |  `OFF`  | Enable Clang sanitizers (tsan is mutually exclusive w.r.t. ubsan and  asan as they don't play well together) |

<!--- @formatter:on --->

## 3. Consuming TypeART
Example using CMake [FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html) for consuming the TypeART runtime library.

```cmake
FetchContent_Declare(
  typeart
  GIT_REPOSITORY https://github.com/tudasc/TypeART
  GIT_TAG v1.9
  GIT_SHALLOW 1
)
FetchContent_MakeAvailable(typeart)

target_link_libraries(my_project_target PRIVATE typeart::Runtime)
```

## References

<table style="border:0px">
<tr>
    <td valign="top"><a name="ref-typeart-2018"></a>[TA18]</td>
    <td>Hück, Alexander and Lehr, Jan-Patrick and Kreutzer, Sebastian and Protze, Joachim and Terboven, Christian and Bischof, Christian and Müller, Matthias S.
    <a href=https://doi.org/10.1109/Correctness.2018.00011>
    Compiler-aided type tracking for correctness checking of MPI applications</a>.
    In <i>2nd International Workshop on Software Correctness for HPC Applications (Correctness)</i>,
    pages 51–58. IEEE, 2018.</td>
</tr>
<tr>
    <td valign="top"><a name="ref-typeart-2020"></a>[TA20]</td>
    <td>Hück, Alexander and Protze, Joachim and Lehr, Jan-Patrick and Terboven, Christian and Bischof, Christian and Müller, Matthias S.
    <a href=https://doi.org/10.1109/Correctness51934.2020.00010>
    Towards compiler-aided correctness checking of adjoint MPI applications</a>.
    In <i>4th International Workshop on Software Correctness for HPC Applications (Correctness)</i>,
    pages 40–48. IEEE/ACM, 2020.</td>
</tr>
<tr>
    <td valign="top"><a name="ref-typeart-2022"></a>[TA22]</td>
    <td>Hück, Alexander and Kreutzer, Sebastian and Protze, Joachim and Lehr, Jan-Patrick and Bischof, Christian and Terboven, Christian and Müller, Matthias S.
    <a href=https://doi.org/10.1109/MITP.2021.3093949>
    Compiler-Aided Type Correctness of Hybrid MPI-OpenMP Applications</a>.
    In <i>IT Professional</i>, vol. 24, no. 2, pages 45–51. IEEE, 2022.</td>
</tr>
<tr>
    <td valign="top"><a name="ref-must-2013"></a>[MU13]</td>
    <td>Hilbrich, Tobias and Protze, Joachim and Schulz, Martin and de Supinski, Bronis R. and Müller, Matthias S.
    <a href=https://doi.org/10.3233/SPR-130368>
    MPI Runtime Error Detection with MUST: Advances in Deadlock Detection</a>.
    In <i>Scientific Programming</i>, vol. 21, no. 3-4,
    pages 109–121, 2013.</td>
</tr>
</table>

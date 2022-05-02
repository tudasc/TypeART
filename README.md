# TypeART &middot; [![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) ![](https://github.com/tudasc/TypeART/workflows/TypeART-CI/badge.svg?branch=master) ![](https://github.com/tudasc/TypeART/workflows/TypeART-CI-ext/badge.svg?branch=master) [![Coverage Status](https://coveralls.io/repos/github/tudasc/TypeART/badge.svg?branch=master)](https://coveralls.io/github/tudasc/TypeART)

## What is TypeART?

TypeART \[[TA18](#ref-typeart-2018); [TA20](#ref-typeart-2020)\] is a type and memory allocation tracking sanitizer
based on the [LLVM](https://llvm.org) compiler toolchain for C/C++ (OpenMP) codes. It consists of an LLVM compiler pass
plugin for instrumentation, and a corresponding runtime to track memory allocations during the execution of a target
program.

TypeART instruments heap, stack and global variable allocations with a callback to our runtime. The callback consists of
(1) the memory address, (2) the type-layout information of the allocation (built-ins, user-defined structs etc.) and (3)
number of elements.

## Why use it?

TypeART provides type-related information of allocations in your program to help verify some property, and to help
generate diagnostics if it doesn't hold.

For instance, low-level C-language APIs use `void`-pointers as generic types. Often, the user must specify its type and
length manually. This can be error-prone. Examples for type unsafe APIs include the Message-Passing Interface (MPI),
checkpointing libraries and numeric solver libraries. With TypeART, it is straightforward to verify that a `void`
-pointer argument to an API is, e.g., a type `T` array with length `n`.

### Use Case: MUST - A dynamic MPI correctness checker

MUST \[[MU13](#ref-must-2013)\] is a dynamic MPI correctness checker to, e.g., detect deadlocks or a mismatch of MPI
datatypes of the sending and receiving process, see its [project page](https://www.hpc.rwth-aachen.de/must/).

MUST relies on intercepting MPI calls for its analysis. As a consequence, though, MUST is unaware of the *effective*
type of the allocated `void*` buffers used for the low-level MPI API. To that end, TypeART was developed to track
memory (de-)allocation relevant to MPI communication. With TypeART, MUST can check for type compatibility between the
type-less MPI communication buffer and the declared MPI datatype.

#### Type checking for MPI calls

Consider the MPI function `MPI_Send(const void* buffer, int count, MPI_Datatype datatype, ...)`. Without TypeART, MUST
cannot check (1) if the `buffer` argument is compatible with the declared `MPI_Dataype` and (2) if the `count` argument
exceeds the `buffer` allocation size:

```c
// TypeART tracks this allocation (memory address, type and size):
double* array = (double*) malloc(length*sizeof(double));
// MUST intercepts this MPI call, and asks TypeARTs runtime library for type information:
//   1. Is the first argument of type double (due to MPI_DOUBLE)?
//   2. Is the allocation at least of size *length*? 
MPI_Send((void*) array, length, MPI_DOUBLE, ...)
```

MUST and TypeART also handle MPI [derived datatypes](https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report/node77.htm)
with complex underlying data structures, see our [MPI Demo](#13-example-mpi-demo). For more details, see
our [publications](#references), or download the current release of MUST (1.8 or higher has TypeART integrated) on
its [project page](https://itc.rwth-aachen.de/must/).

## Table of Contents

* [1. Using TypeART](#1-using-typeart)
    * [1.1 Compiling a target code](#11-compiling-a-target-code)
        * [1.1.1 Building with TypeART](#111-building-with-typeart)
        * [1.1.2 Options for TypeART passes](#112-options-for-typeart-passes)
        * [1.1.3 Serialized type information](#113-serialized-type-information)
        * [1.1.4 Filtering allocations](#114-filtering-allocations)
    * [1.2 Executing an instrumented target code](#12-executing-an-instrumented-target-code)
    * [1.3 Example: MPI demo](#13-example-mpi-demo)
* [2. Building TypeART](#2-building-typeart)
    * [2.1 Optional software requirements](#21-optional-software-requirements)
    * [2.2 Building](#22-building)
        * [2.2.1 CMake configuration: Options for users](#221-cmake-configuration-options-for-users)
* [References](#references)

## 1. Using TypeART

Making use of TypeART consists of two phases:

1. Compile your code with Clang/LLVM (version >= 10) using the TypeART LLVM pass plugins to (1) serialize static type
   information to a file and (2) instrument all relevant allocations. See [Section 1.1](#11-compiling-a-target-code).
2. Execute the target program with a runtime library (a *client* based on the TypeART runtime) to accept the callbacks
   to do some useful analysis with our interface based on the static type information.
   See [Section 1.2](#12-executing-an-instrumented-target-code).

### 1.1 Compiling a target code

Our LLVM compiler pass plugins instrument allocations and also serialize the static type layouts of these allocations to
a yaml file (default name `types.yaml`). To that end, we provide compiler [wrapper scripts](scripts/typeart-wrapper.in)
around Clang and MPI to apply TypeART in the `bin` folder of the TypeART installation prefix. By default, the wrappers
instrument heap, stack and global allocations. The MPI-wrappers also filter allocations that are not passed to an MPI
call, see [Section 1.1.4](#114-filtering-allocations).

*Note*: Currently, the compilation process has to be serialied, e.g., `make -j 1`, due to extraction and consistency of
type information per translation unit.

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

In particular, the wrapper script does the following:

1. Compile the code down to LLVM IR, and pipe the output to the LLVM `opt` tool. (Keeping your original compile flags)
2. Apply heap instrumentation with TypeART through `opt`.
3. Optimize the code with your `-Ox` flag using `opt`.
4. Apply stack and global instrumentation with TypeART through `opt`.
5. Pipe the final output to LLVM `llc` to generate the final object file.
6. Finally, link the TypeART runtime library using your linker flags.

*Note*: We instrument heap allocations before any optimization, as the compiler may throw out type information of these
allocations (for optimization reasons).

##### Wrapper usage in CMake build systems

For plain Makefiles, the wrapper replaces the GCC/Clang compiler variables, e.g., `CC` or `MPICC`. For CMake, during the
configuration, it is advised to disable the wrapper temporarily. This is due to CMake executing internal compiler
checks, where we do not need TypeART instrumentation:

```shell
# Temporarily disable wrapper with environment flag TYPEART_WRAPPER=OFF for configuration:
$> TYPEART_WRAPPER=OFF cmake -B build -DCMAKE_C_COMPILER=*TypeART bin*/typeart-clang 
# Compile with TypeART now:
$> cmake --build build --target install
```

##### MPI wrapper generation

For MPI, we rely on detecting the vendor to generate wrappers with appropriate environment variables to force the use of
the Clang/LLVM compiler.
We support detection for OpenMPI, Intel MPI and MPICH based on `mpi.h` symbols, and use the following flags for setting
the Clang
compiler:

| Vendor    | Symbol        | C compiler env. var  | C++ compiler env. var  |
|-----------|---------------|----------------------|------------------------|
| Open MPI  | OPEN_MPI      | OMPI_CC              | OMPI_CXX               |
| Intel MPI | I_MPI_VERSION | I_MPI_CC             | I_MPI_CXX              |
| MPICH     | MPICH_NAME    | MPICH_CC             | MPICH_CXX              |

##### Internal wrapper invocation

For reference, the wrapper script executes the following (pseudo):

```shell
# Compile: 1.Code-To-LLVM | 2.TypeART_HEAP | 3.Optimize | 4.TypeART_Stack | 5.Object-file 
$> clang++ $(COMPILE_FLAGS) $(EMIT_LLVM_IR_FLAGS) code.cpp | opt $(TYPEART_PLUGIN) $(HEAP_ONLY_FLAGS) | opt -O2 -S | opt $(TYPEART_PLUGIN) $(STACK_ONLY_FLAGS) | llc $(TO_OBJECT_FILE)
# Link:
$> clang++ $(LINK_FLAGS) -L$(TYPEART_LIBPATH) -ltypeartRuntime code.o -o binary
```

#### 1.1.2 Options for TypeART passes

For modification of the pass behavior, we provide several options.
<!--- @formatter:off --->
| Flag                        |   Default    | Description                                                                                                                                        |
|-----------------------------|:------------:|----------------------------------------------------------------------------------------------------------------------------------------------------|
| `typeart`                   |      -       | Invoke TypeART pass through LLVM `opt`                                                                                                             |
| `typeart-types`           | `types.yaml` | Serialized type layout information of user-defined types. File location and name can also be controlled with the env variable `TYPEART_TYPE_FILE`. |
| `typeart-heap`              |    `true`    | Instrument heap allocations                                                                                                                        |
| `typeart-stack`            |   `false`    | Instrument stack and global allocations. Enables instrumentation of global allocations.                                                            |
| `typeart-global`    |   `false`    | Instrument global allocations (see --typeart-stack).                                                                                               |
| `typeart-stats`             |   `false`    | Show instrumentation statistic counters                                                                                                            |
| `typeart-call-filter`               |   `false`    | Filter stack and global allocations. See also [Section 1.1.4](#114-filtering-allocations)                                                          |
| `typeart-call-filter-str`           |   `*MPI_*`   | Filter string target (glob string)                                                                                                                 |
| `typeart-filter-pointer-alloca` |    `true`    | Filter stack alloca of pointers (typically generated by LLVM for references of stack vars)                                                         |

<!--- @formatter:on --->

##### Example invocations

###### Pre-requisites

1. Loading TypeART plugins with `opt`:
    ```shell
    TYPEART_PLUGIN=-load $(PLUGIN_PATH)/typeartTransformPass.so
    ```
2. Input of `opt` is LLVM IR, e.g.:
    ```shell
    # Pipe LLVM IR to console:
    $> clang++ -O1 -g -Xclang -disable-llvm-passes -S -emit-llvm -o - example.cpp
    ```

###### Examples

- Heap-only instrumentation (with stats):
    ```shell
    opt $(TYPEART_PLUGIN) -typeart -typeart-stats
    ```
- Stack- and global-only instrumentation (*no* stats):
    ```shell
    opt $(TYPEART_PLUGIN) -typeart -typeart-heap=true -typeart-stack
    ```
- Stack- and global-only instrumentation (with default filtering for MPI):
    ```shell
    opt $(TYPEART_PLUGIN) -typeart -typeart-heap=false -typeart-stack -typeart-call-filter
    ```
- Filtering w.r.t. non-standard target API:
    ```shell
    opt $(TYPEART_PLUGIN) -typeart -typeart-heap=false -typeart-stack -typeart-call-filter -typeart-call-filter-str=MY_API*
    ```
- Combined instrumentation (with filtering):
    ```shell
    opt $(TYPEART_PLUGIN) -typeart -typeart-stack -typeart-call-filter
    ```

#### 1.1.3 Serialized type information

After instrumentation, the file `types.yaml` contains the static type information. Each user-defined type layout is
extracted and an integer `type-id` is attached to it. Built-in types (e.g., float) have pre-defined ids and byte
layouts.

The TypeART instrumentation callbacks use the `type-id`. The runtime library correlates the allocation with the
respective type (and layout) during execution. Consider the following struct:

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
  name: struct.s1_t
  extent: 16         // byte size
  member_count: 2
  offsets: [ 0, 8 ]  // byte offsets from struct start
  types: [ 0, 10 ]   // member type-ids (0->char, 10->pointer)
  sizes: [ 3, 1 ]    // member (array) length
```

<!--- @formatter:on --->

#### 1.1.4 Filtering allocations

To improve performance, a translation unit-local (TU) data-flow filter for global and stack variables exist. It follows
the LLVM IR `use-def` chain. If the allocation provably never reaches the target API, it can be filtered. Otherwise, it
is instrumented.

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
$> export TYPEART_TYPE_FILE=/shared/types.yaml
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

TypeART requires LLVM version >= 10 and CMake version >= 3.20.

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
| Option | Default | Description |
| --- | :---: | --- |
| `TYPEART_MPI_WRAPPER` | `ON` | Install TypeART MPI wrapper (mpic, mpic++). Requires MPI. |
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

| Option | Default | Description |
| --- | :---: | --- |
| `TYPEART_DISABLE_THREAD_SAFETY` | `OFF` | Disable thread safety of runtime |
| `TYPEART_SAFEPTR` | `OFF` | Instead of a mutex, use a special data structure wrapper for concurrency, see [object_threadsafe](https://github.com/AlexeyAB/object_threadsafe) |

<!--- @formatter:on --->

##### LLVM passes

<!--- @formatter:off --->

| Option | Default | Description |
| --- | :---: | --- |
| `TYPEART_SHOW_STATS` | `ON` | Passes show compile-time summary w.r.t. allocations counts |
| `TYPEART_MPI_INTERCEPT_LIB` | `ON` | Library to intercept MPI calls by preloading and check whether TypeART tracks the buffer pointer |
| `TYPEART_MPI_LOGGER` | `ON` | Enable better logging support in MPI execution context |
| `TYPEART_LOG_LEVEL` | `0` | Granularity of pass logger. 3 is most verbose, 0 is least |

<!--- @formatter:on --->

##### Testing

<!--- @formatter:off --->

| Option | Default | Description                                                                                                  |
| --- | :---: |--------------------------------------------------------------------------------------------------------------|
| `TYPEART_TEST_CONFIG` | `OFF` | Enable testing, and set (force) logging levels to appropriate levels for test runner to succeed              |
| `TYPEART_CODE_COVERAGE` | `OFF` | Enable code coverage statistics using LCOV 1.14 and genhtml (gcovr optional)                                 |
| `TYPEART_LLVM_CODE_COVERAGE` | `OFF` | Enable llvm-cov code coverage statistics (llvm-cov and llvm-profdata  required)                              |
| `TYPEART_ASAN, TSAN, UBSAN` | `OFF` | Enable Clang sanitizers (tsan is mutually exclusive w.r.t. ubsan and  asan as they don't play well together) |

<!--- @formatter:on --->

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
    <td valign="top"><a name="ref-must-2013"></a>[MU13]</td>
    <td>Hilbrich, Tobias and Protze, Joachim and Schulz, Martin and de Supinski, Bronis R. and Müller, Matthias S.
    <a href=https://doi.org/10.3233/SPR-130368>
    MPI Runtime Error Detection with MUST: Advances in Deadlock Detection</a>.
    In <i>Scientific Programming</i>, vol. 21, no. 3-4,
    pages 109–121, 2013.</td>
</tr>
</table>
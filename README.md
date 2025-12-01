# TypeART &middot; [![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) ![](https://github.com/tudasc/TypeART/actions/workflows/basic-ci.yml/badge.svg?branch=master) ![](https://github.com/tudasc/TypeART/actions/workflows/ext-ci.yml/badge.svg?branch=master) [![Coverage Status](https://coveralls.io/repos/github/tudasc/TypeART/badge.svg?branch=master)](https://coveralls.io/github/tudasc/TypeART?branch=master)

## What is TypeART?

TypeART \[[TA18](#ref-typeart-2018); [TA20](#ref-typeart-2020); [TA22](#ref-typeart-2022); [TA24](#ref-typeart-2024)\] is a type and memory
allocation tracking sanitizer based on the [LLVM](https://llvm.org) compiler toolchain for C/C++ (OpenMP) codes. It pairs a compiler plugin (for instrumentation) with a runtime library to track memory type, size, and location of heap, stack and global allocations.


## Why use it?

Low-level C APIs often rely on `void*` pointers for generic types, requiring users to specify type and size manually, a process prone to errors. Examples of type-unsafe APIs include the Message-Passing Interface (MPI), checkpointing libraries, and numeric solver libraries. TypeART facilitates verification by ensuring, for example, that a `void*` argument corresponds to an array of expected type `T` with length `n`.


### Use Case: MUST - A dynamic MPI correctness checker

MUST \[[MU13](#ref-must-2013)\], a dynamic MPI correctness checker, detects issues like deadlocks or mismatched MPI datatypes. For more details, visit its [project page](https://www.hpc.rwth-aachen.de/must/).

MUST intercepts MPI calls for analysis but cannot deduce the *effective* type of `void*` buffers in MPI APIs. TypeART addresses this by tracking memory allocations relevant to MPI communication in user code, allowing MUST to validate type compatibility between MPI buffers and declared datatypes.

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
    * [1.2 Executing an instrumented target code](#12-executing-an-instrumented-target-code)
    * [1.3 Example: MPI demo](#13-example-mpi-demo)
* [2. TypeART compiler pass](#2-typeart-compiler-pass)
    * [2.1 Options for controlling the TypeART pass](#21-options-for-controlling-the-typeart-pass)
    * [2.2 Serialized type information](#22-serialized-type-information)
    * [2.3 Filtering allocations](#23-filtering-allocations)
* [3. Building TypeART](#3-building-typeart)
    * [3.1 Optional software requirements](#31-optional-software-requirements)
    * [3.2 Building](#32-building)
    * [3.3 CMake configuration: Options for users](#33-cmake-configuration-options-for-users)
* [4. Consuming TypeART](#4-consuming-typeart)
* [References](#references)

## 1. Using TypeART

Using TypeART involves two phases:

1. Compilation, see [Section 1.1](#11-compiling-a-target-code): Compile code with Clang/LLVM using the TypeART LLVM pass plugin via the compiler wrapper script. The plugin (1) serializes static type information and (2) instruments relevant allocations.
2. Execution, see [Section 1.2](#12-executing-an-instrumented-target-code): Run the instrumented program. The TypeART runtime tracks all memory allocations. Clients can query the runtime for type information regarding a memory pointer at relevant points during program execution.

```
+----Compiler----+         +-----------------------------------+
| typeart-mpicc  +----+--->|  TypeART-instrumented Application |
+----------------+    |    +--+-----+-------------------+------+
        ^           Static    |     |                   |       
        |            Type     v     v                   v       
   +----+----+       Info   Alloc/Free           Intercepted API
   | Sources |        |    +-----------+         +-------------+
   +---------+        |    |  TypeART  |+--------+ Correctness |
                      +--->|  Runtime  ||  Query |    Tool     |
                           |           |+------->| (ex. MUST)  |
                           +-----------+         +-------------+
```


### 1.1 Compiling a target code

The TypeART LLVM compiler pass instruments allocations and serializes static type layouts. Compiler wrapper scripts are provided (available in the `bin` directory of the installation) for Clang and MPI. By default, these wrappers instrument heap, stack, and global allocations. MPI wrappers additionally filter allocations unrelated to MPI calls (see [Section 2.3](#23-filtering-allocations)).

#### Building with TypeART

Replace the compiler variable as follows:

| Variable | TypeART Wrapper   | Equivalent to |
|----------|-------------------|---------------|
| `CXX`    | `typeart-clang++` | `clang++`     |
| `CC`     | `typeart-clang`   | `clang`       |
| `MPICC`  | `typeart-mpicc`   | `mpicc`       |
| `MPICXX` | `typeart-mpic++`  | `mpic++`      |

The wrappers handle the LLVM pass injection and linking:

```shell
# Compile, replace direct clang++ call with wrapper of the TypeART installation:
$> typeart-clang++ -O2 $(COMPILE_FLAGS) -c code.cpp -o code.o
# Link, also with the wrapper:
$> typeart-clang++ $(LINK_FLAGS) code.o -o binary
```

##### CMake projects

When using CMake, disable the wrapper during configuration (to pass internal compiler checks) but enable it for the build step.

```shell
# Temporarily disable wrapper with environment flag TYPEART_WRAPPER=OFF for configuration:
$> TYPEART_WRAPPER=OFF cmake -B build -DCMAKE_C_COMPILER=/path/to/typeart-clang 
# Compile with typeart-clang:
$> cmake --build build --target install
```

### 1.2 Executing an instrumented target code

Execute the target binary directly.

```shell
# Ensure the TypeART runtime is in the library path:
$> env LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(TYPEART_LIBPATH) ./binary
```


### 1.3 Example: MPI demo

The folder [demo](demo) contains an example of MPI-related type errors that can be detected using TypeART. The target code is instrumented with TypeART, and executed by preloading the MPI-related check library implemented
in [tool.c](demo/tool.c). The tool library uses the TypeART [runtime query interface](lib/runtime/RuntimeInterface.h).
It overloads the required MPI calls and checks that the passed `void*` buffer corresponds to the MPI derived datatype.

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


## 2 TypeART compiler pass

### 2.1 Options for controlling the TypeART pass

Pass behavior is configured via the environment flags listed below. The TypeART pass prioritizes environment flags (if set) over default configuration options.

Specifically, `TYPEART_OPTIONS` can globally modify the TypeART pass (stack/heap specific options exist). The format requires option names separated by a semicolon, e.g., `TYPEART_OPTIONS="filter-glob=API_*;no-stats"` sets the filter glob target to `API_*` and deactivates stats printing. Prepending `no-` to boolean flags sets them to `false`.

**Note**: Single environment options take precedence over `TYPEART_OPTIONS`.

<!--- @formatter:off --->

| Env. variable                              | Option name                        |    Default value     | Description                                                                                                                                                |
|:-------------------------------------------|------------------------------------|:--------------------:|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `TYPEART_OPTIONS`                          |                                    |                      | Set multiple options at once, separated by `;`.                                                                                                            |
| `TYPEART_OPTIONS_STACK`                    |                                    |                      | Same as above for stack phase only.                                                                                                                        |
| `TYPEART_OPTIONS_HEAP`                     |                                    |                      | Same as above for heap phase only.                                                                                                                         |
| `TYPEART_TYPES`                            | `types`                            | `typeart-types.yaml` | Serialized type layout information of user-defined types. File location and name can also be controlled with the env variable `TYPEART_TYPES`.             |
| `TYPEART_HEAP`                             | `heap`                             |        `true`        | Instrument heap allocations                                                                                                                                |
| `TYPEART_STACK`                            | `stack`                            |       `false`        | Instrument stack and global allocations. Enables instrumentation of global allocations.                                                                    |
| `TYPEART_STACK_LIFETIME`                   | `stack-lifetime`                   |        `true`        | Instrument stack `llvm.lifetime.start` instead of `alloca` directly                                                                                        |
| `TYPEART_GLOBAL`                           | `global`                           |       `false`        | Instrument global allocations (see stack).                                                                                                                 |
| `TYPEART_TYPEGEN`                          | `typegen`                          |       `dimeta`       | Values: `dimeta`, `ir`. How serializing of type information is done, see [Section 2.2](#22-serialized-type-information).                                   |
| `TYPEART_TYPE_SERIALIZATION`               | `type-serialization`               |       `hybrid`       | Values: `file`, `hybrid`, `inline`. How type information are stored (in the executable or externally), see [Section 2.2](#22-serialized-type-information). |
| `TYPEART_STATS`                            | `stats`                            |       `false`        | Show instrumentation statistic counters                                                                                                                    |
| `TYPEART_FILTER`                           | `filter`                           |       `false`        | Filter stack and global allocations. See also [Section 2.3](#23-filtering-allocations)                                                                     |
| `TYPEART_FILTER_IMPLEMENTATION`            | `filter-implementation`            |        `std`         | Values: `std`, `none`. See also [Section 2.3](#23-filtering-allocations)                                                                                   |
| `TYPEART_FILTER_GLOB`                      | `filter-glob`                      |       `*MPI_*`       | Filter API string target (glob string)                                                                                                                     |
| `TYPEART_FILTER_GLOB_DEEP`                 | `filter-glob-deep`                 |       `MPI_*`        | Filter values based on specific API: Values passed as ptr are correlated when string matched.                                                              |
| `TYPEART_ANALYSIS_FILTER_GLOBAL`           | `analysis-filter-global`           |        `true`        | Filter global alloca based on heuristics                                                                                                                   |
| `TYPEART_ANALYSIS_FILTER_HEAP_ALLOCA`      | `analysis-filter-heap-alloca`      |        `true`        | Filter stack alloca that have a store instruction from a heap allocation                                                                                   |
| `TYPEART_ANALYSTS_FILTER_NON_ARRAY_ALLOCA` | `analysis-filter-non-array-alloca` |       `false`        | Filter scalar valued allocas                                                                                                                               |
| `TYPEART_ANALYSIS_FILTER_POINTER_ALLOCA`   | `analysis-filter-pointer-alloca`   |        `true`        | Filter allocas of pointer types                                                                                                                            |

Additionally, there are two debug environment flags for dumping the LLVM IR per phase (pre heap, heap, opt, stack) to a set of files.

| Env. variable                   | Description                                                                                                                     |
|:--------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| `TYPEART_WRAPPER_EMIT_IR`       | If set, the compiler wrapper will create 4 files for each TypeART phase with the file pattern `${source_basename}_heap.ll` etc. |
| `TYPEART_PASS_INTERNAL_EMIT_IR` | Internal pass use only. Toggled by wrapper.                                                                                     |

<!--- @formatter:on --->


### 2.2 Serialized type information

TypeART uses either the [LLVM IR type system](https://llvm.org/docs/LangRef.html#type-system) (`typegen=ir`) or the external library [llvm-dimeta](https://github.com/ahueck/llvm-dimeta) (`typegen=dimeta`), which extracts type information using [LLVM debug metadata](https://llvm.org/docs/SourceLevelDebugging.html). The latter is the default; the former is compatible only with LLVM 14.

The layout is serialized either as a global variable inside each translation unit (`type-serialization=hybrid` or `inline`) or via an external YAML file (`type-serialization=file`).

**Note**: In `file` mode, compilation must be serialized (e.g., `make -j 1`) to ensure consistent type information across translation units.


#### 2.2.1 Hybrid and Inline serialization

Type serialization for each user-defined type (mode `hybrid`) or *all* types (mode `inline`) are stored as (constant) globals with the following format:

```c
struct GlobalTypeInfo {
  std::int32_t type_id;
  const std::uint32_t extent;
  const std::uint16_t num_members;
  const std::uint16_t flag;
  const char* type_name;
  const std::uint16_t* offsets;
  const std::uint16_t* array_sizes;
  const GlobalTypeInfo** member_types;
};
```

Each type is registered at startup with the TypeART runtime using the callback `void __typeart_register_type(const void* type_ptr);`. This adds the type information to the type database (for user queries) and assigns a unique `type-id`.
Each user-defined type layout is assigned a unique integer `type-id` starting at 256. Built-in types (e.g., `float`) use predefined type-ids (\< 256) and byte layouts. The runtime library correlates the allocation with the respective type (and layout) during execution via the `type-id`.


#### 2.2.2 File-based serialization
After instrumentation, the file `typeart-types.yaml` (`env TYPEART_TYPES`) contains the static type information. Each user-defined type layout is
extracted and an integer `type-id` is attached to it (similarly to hybrid and inline serialization).
For example, consider the following C struct:

```c
struct s1_t {
  char a[3];
  struct s1_t* b;
}
```

The TypeART pass may write a `typeart-types.yaml` file with the following content:
<!--- @formatter:off --->

```yaml
- id: 256           // struct type-id
  name: s1_t
  extent: 16        // size in bytes
  member_count: 2
  offsets: [ 0, 8 ] // byte offsets from struct start
  types: [ 5, 1 ]   // member type-ids (5->char, 1->pointer)
  sizes: [ 3, 1 ]   // member (array) length
```

<!--- @formatter:on --->

Executing a target binary requires access to the `typeart-types.yaml` file to correlate the type-id with actual type layouts. Specify the path using the environment variable `TYPEART_TYPES`:

```bash
$> export TYPEART_TYPES=/path/to/typeart-types.yaml
# If the TypeART runtime is not resolved, LD_LIBRARY_PATH is set:
$> env LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(TYPEART_LIBPATH) ./binary
```

#### 2.2.3 Side note: Limitations of LLVM IR Type System

The list of supported built-in type-ids is defined in [TypeInterface.h](lib/typelib/TypeInterface.h) and reflects the types that TypeART can represent with **LLVM Debug Metadata**.
In contrast, when using **LLVM IR Type System**, certain constraints are imposed. For instance, C/C++ types like unsigned integers are unsupported (and represented like signed integers). 


### 2.3 Filtering allocations

To improve performance, a translation unit-local (TU) data-flow filter for global and stack variables exist. It follows the LLVM IR use-def chain. If the allocation provably never reaches the target API, it can be filtered. Otherwise, it is instrumented. Use the option `filter` to enable filtering and `filter-glob=<target API glob>` (default: `*MPI_*`) to specify the API.

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

1.  `a` is filtered because the aliasing pointer `x` is never part of an MPI call.
2.  `b` is instrumented because the aliasing pointer `y` is part of an MPI call.
3.  `c` is instrumented because the body of `foo_bar` cannot be reasoned about.


## 3. Building TypeART

TypeART supports LLVM version 14, 18-21, and CMake version >= 3.20.

### 3.1 Optional software requirements

- MPI library: (soft requirement) Needed for the MPI compiler wrappers, tests, the [demo](demo),
  our [MPI interceptor library](lib/mpi_interceptor), and for logging with our TypeART runtime library within an MPI
  target application.
- OpenMP-enabled Clang compiler: Needed for some tests.

Other smaller, external dependencies are defined within the [externals folder](externals) (depending on configuration
options), see [Section 3.3 (Runtime)](#33-cmake-configuration-options-for-users). They are automatically downloaded
during configuration time.

### 3.2 Building

TypeART uses CMake to build, cf. [GitHub CI build file](.github/workflows/basic-ci.yml) for a complete recipe to build.
Example build recipe (debug build, installs to default prefix
`${typeart_SOURCE_DIR}/install/typeart`)

```sh
$> git clone https://github.com/tudasc/TypeART
$> cd TypeART
$> cmake -B build
$> cmake --build build --target install --parallel
```

### 3.3 CMake configuration: Options for users

##### Binaries (scripts)

<!--- @formatter:off --->

| Option                       | Default | Description                                                                      |
|------------------------------|:-------:|----------------------------------------------------------------------------------|
| `TYPEART_MPI_WRAPPER`        |  `ON`   | Install TypeART MPI wrapper (mpic, mpic++). Requires MPI.                        |
| `TYPEART_USE_LEGACY_WRAPPER` |  `OFF`  | Use legacy wrapper invoking opt/llc directly instead of Clang's `-fpass-plugin`. |


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

| Option                          | Default | Description                                                                                                                                      |
|---------------------------------|:-------:|--------------------------------------------------------------------------------------------------------------------------------------------------|
| `TYPEART_DISABLE_THREAD_SAFETY` |  `OFF`  | Disable thread safety of runtime                                                                                                                 |
| `TYPEART_SAFEPTR`               |  `OFF`  | Instead of a mutex, use a special data structure wrapper for concurrency, see [object_threadsafe](https://github.com/AlexeyAB/object_threadsafe) |

<!--- @formatter:on --->

##### LLVM pass

<!--- @formatter:off --->

| Option                      | Default | Description                                                                                      |
|-----------------------------|:-------:|--------------------------------------------------------------------------------------------------|
| `TYPEART_SHOW_STATS`        |  `ON`   | Passes show compile-time summary w.r.t. allocations counts                                       |
| `TYPEART_MPI_INTERCEPT_LIB` |  `ON`   | Library to intercept MPI calls by preloading and check whether TypeART tracks the buffer pointer |
| `TYPEART_MPI_LOGGER`        |  `ON`   | Enable better logging support in MPI execution context                                           |
| `TYPEART_LOG_LEVEL`         |   `0`   | Granularity of pass logger. 3 is most verbose, 0 is least                                        |

<!--- @formatter:on --->

##### Testing

<!--- @formatter:off --->

| Option                       | Default | Description                                                                                                  |
|------------------------------|:-------:|--------------------------------------------------------------------------------------------------------------|
| `TYPEART_TEST_CONFIG`        |  `OFF`  | Enable testing, and set (force) logging levels to appropriate levels for test runner to succeed              |
| `TYPEART_CODE_COVERAGE`      |  `OFF`  | Enable code coverage statistics using LCOV 1.14 and genhtml (gcovr optional)                                 |
| `TYPEART_LLVM_CODE_COVERAGE` |  `OFF`  | Enable llvm-cov code coverage statistics (llvm-cov and llvm-profdata  required)                              |
| `TYPEART_ASAN, TSAN, UBSAN`  |  `OFF`  | Enable Clang sanitizers (tsan is mutually exclusive w.r.t. ubsan and  asan as they don't play well together) |

<!--- @formatter:on --->

#### 3.3.1 CMake Internals

##### MPI wrapper generation

The wrappers `typeart-mpicc` and `typeart-mpic++` are generated for compiling MPI codes with TypeART.
The build system detects the vendor to generate wrappers with appropriate environment variables that force the use of the Clang/LLVM compiler.
Detection is supported for OpenMPI, Intel MPI, and MPICH based on `mpi.h` symbols. The following flags are used to set the Clang compiler:

| Vendor    | Symbol        | C compiler env. var | C++ compiler env. var |
|-----------|---------------|---------------------|-----------------------|
| Open MPI  | OPEN_MPI      | OMPI_CC             | OMPI_CXX              |
| Intel MPI | I_MPI_VERSION | I_MPI_CC            | I_MPI_CXX             |
| MPICH     | MPICH_NAME    | MPICH_CC            | MPICH_CXX             |


## 4. Consuming TypeART
Example using CMake [FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html) for consuming the TypeART runtime library.

```cmake
FetchContent_Declare(
  typeart
  GIT_REPOSITORY https://github.com/tudasc/TypeART
  GIT_TAG v2.2
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
    "Compiler-aided type tracking for correctness checking of MPI applications."
    In <i>2nd International Workshop on Software Correctness for HPC Applications (Correctness)</i>,
    pages 51–58. IEEE, 2018. DOI: <a href=https://doi.org/10.1109/Correctness.2018.00011>10.1109/Correctness.2018.00011</a></td>
</tr>
<tr>
    <td valign="top"><a name="ref-typeart-2020"></a>[TA20]</td>
    <td>Hück, Alexander and Protze, Joachim and Lehr, Jan-Patrick and Terboven, Christian and Bischof, Christian and Müller, Matthias S.
    "Towards compiler-aided correctness checking of adjoint MPI applications."
    In <i>4th International Workshop on Software Correctness for HPC Applications (Correctness)</i>,
    pages 40–48. IEEE/ACM, 2020. DOI: <a href=https://doi.org/10.1109/Correctness51934.2020.00010>10.1109/Correctness51934.2020.00010</a></td>
</tr>
<tr>
    <td valign="top"><a name="ref-typeart-2022"></a>[TA22]</td>
    <td>Hück, Alexander and Kreutzer, Sebastian and Protze, Joachim and Lehr, Jan-Patrick and Bischof, Christian and Terboven, Christian and Müller, Matthias S.
    "Compiler-Aided Type Correctness of Hybrid MPI-OpenMP Applications."
    In <i>IT Professional</i>, vol. 24, no. 2, pages 45–51. IEEE, 2022. DOI: <a href=https://doi.org/10.1109/MITP.2021.3093949>10.1109/MITP.2021.3093949</a></td>
</tr>
<tr>
    <td valign="top"><a name="ref-typeart-2024"></a>[TA24]</td>
    <td>Hück, Alexander and Ziegler, Tim and Schwitanski, Simon and Jenke, Joachim and Bischof, Christian. <!-- codespell:ignore -->
    "Compiler-Aided Correctness Checking of CUDA-Aware MPI Applications."
    In <i>SC24-W: Workshops of the International Conference for High Performance Computing, Networking, Storage and Analysis</i>, 
    pages 204–213, IEEE/ACM, 2024. DOI: <a href=https://doi.org/10.1109/SCW63240.2024.00032>10.1109/SCW63240.2024.00032</a></td>
</tr>
<tr>
    <td valign="top"><a name="ref-must-2013"></a>[MU13]</td>
    <td>Hilbrich, Tobias and Protze, Joachim and Schulz, Martin and de Supinski, Bronis R. and Müller, Matthias S.
    "MPI Runtime Error Detection with MUST: Advances in Deadlock Detection."
    In <i>Scientific Programming</i>, vol. 21, no. 3-4,
    pages 109–121, 2013. DOI: <a href=https://doi.org/10.3233/SPR-130368>10.3233/SPR-130368</a></td>
</tr>
</table>

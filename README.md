# TypeART &middot; [![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) ![](https://github.com/tudasc/TypeART/workflows/TypeART-CI/badge.svg?branch=master) ![](https://github.com/tudasc/TypeART/workflows/TypeART-CI-ext/badge.svg?branch=master) [![Coverage Status](https://coveralls.io/repos/github/tudasc/TypeART/badge.svg?branch=master)](https://coveralls.io/github/tudasc/TypeART)

## What is TypeART?

TypeART \[[TA18](#ref-typeart-2018); [TA20](#ref-typeart-2020)\] is a type and memory allocation tracking sanitizer
based on the LLVM compiler toolchain. It consists of an LLVM compiler pass plugin for instrumentation and a
corresponding runtime to track memory allocations during the execution of a target program.

TypeART instruments heap, stack and global variable allocations with a callback to our runtime. The callback consists of
(1) the memory address, (2) the type-layout information of the allocation (built-ins, user-defined structs etc.) and (3)
extent of the value. This allows users of our runtime to query detailed type information behind arbitrary memory
locations, as long as they are mapped.

TypeART also works with OpenMP codes and its runtime is thread-safe (since release 1.6).

## Why use it?

Employ TypeART whenever you need type information of allocations in your program to verify some property, and generate
diagnostics if it doesn't hold.

For instance, low-level C-language APIs use `void`-pointers as generic types. Often, the user must specify its type and
length manually. This can be error prone, especially in larger, long-lived project s with changing developers.

With TypeART, it is straightforward to verify that a `void`-pointer argument to an API is, e.g., a type `T` array with
length `n`. Examples for type unsafe APIs include the Message-Passing Interface (MPI), checkpointing libraries and
numeric solver libraries.

### Use Case: MUST - A dynamic MPI correctness checker

MUST \[[MU13](#ref-must-2013)\] is a dynamic MPI correctness checker that is able to, e.g., detect deadlocks or a
mismatch of MPI datatypes of the sending and receiving process, see
its [project page](https://www.hpc.rwth-aachen.de/must/).

MUST relies on intercepting MPI calls for its analysis. As a consequence, though, MUST is unaware of the *effective*
type of the allocated `void*` buffers used for the low-level MPI API. To that end, TypeART was developed to track
memory (de-)allocation relevant to MPI communication. With TypeART, MUST can check for type compatibility between the
type-less communication buffer and the declared MPI datatype.

#### Type checking for MPI calls

Consider the MPI function `MPI_Send(const void* buffer, int count, MPI_Datatype datatype, ...)`. Without TypeART, MUST
cannot check 1) if the `buffer` argument is compatible with the declared `MPI_Dataype` and 2) if `count` does not exceed
the `buffer` allocation size. For instance, if the datatype is `MPI_DOUBLE`, we expect the `buffer` argument to be
a `double*`-array with a minimum size specified by `count`:

```c
// TypeART tracks this allocation (memory address, type and size):
double* array = (double*) malloc(length*sizeof(double));
// MUST intercepts this MPI call, and asks TypeARTs runtime library for type information:
//   1. Is the first argument of type double (due to MPI_DOUBLE)?
//   2. Is the allocation at least of size *length*? 
MPI_Send((void*) array, length, MPI_DOUBLE, ...)
```

MUST and TypeART also handle MPI [derived datatypes](https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report/node77.htm)
with complex underlying datastructures, see [our demo folder](demo). For more details, see our publications below or
download the current release (1.8 or higher) of MUST (with TypeART) on
the [MUST homepage](https://itc.rwth-aachen.de/must/).

## 1. Using TypeART

Making use of TypeART consists of two phases:

1. Compile your code with Clang/LLVM-10 using the TypeART LLVM pass plugins to 1) extract static type information to a
   `yaml`-file and 2) instrument all relevant allocations. Allocations can be filtered if they are never passed to your
   target API, e.g., MPI.
2. Execute the target program with a runtime library (a *client* based on the TypeART runtime) to accept the callbacks
   from the instrumented code and actually do some useful analysis. To that end, the
   interface [RuntimeInterface.h](lib/runtime/RuntimeInterface.h) can be used as a type-information query interface for
   clients.

### 1.1 Compiling a target code

Our LLVM compiler pass plugins instrument allocations and also serialize the static type layouts of these allocations to
a yaml file (default name `types.yaml`).

#### 1.1.0 Caveats

Unfortunately, applying TypeART is not yet user-friendly. You must change the build system to accommodate the use of the
LLVM optimizer tool `opt`. It loads and applies our compiler pass extension. A more convenient way through Clang
compiler wrappers (like [mpicc](https://www.open-mpi.org/doc/v4.1/man1/mpicc.1.php)) are planned for a future release.

#### 1.1.1 Building with TypeART

A typical compile invocation may first compile code to object files and then link with any libraries, e.g.:

```shell
# Compile:
$> clang++ -O2 $(COMPILE_FLAGS) -c code.cpp -o code.o
# Link:
$> clang++ $(LINK_FLAGS) code.o -o binary
```

With TypeART, the recipe needs to be changed, as we rely on the LLVM `opt` (optimizer) tool to load and apply our
TypeART passes to a target code based on the LLVM intermediate representation. To that end, the following steps are
currently needed:

1. Compile the code down to LLVM IR, and pipe the output to the LLVM `opt` tool. (Keeping your original compile flags)
2. Apply heap instrumentation with TypeART through `opt`.
3. Optimize the code with -Ox using `opt`.
4. Apply stack and global instrumentation with TypeART through `opt`.
5. Pipe the final output to LLVM `llc` to generate the final object file.
6. Subsequently, the TypeART runtime library is linked (for the added instrumentation callbacks).

Note: We instrument heap allocations before any optimization, as the compiler may throw out related type information of
these allocations (for optimization reasons).

```shell
# Compile: 1.Code-To-LLVM | 2.TypeART_HEAP | 3.Optimize | 4.TypeART_Stack | 5.Object-file 
$> clang++ $(COMPILE_FLAGS) $(EMIT_LLVM_IR_FLAGS) code.cpp | opt $(TYPEART_PLUGIN) $(HEAP_ONLY_FLAGS) | opt -O2 -S | opt $(TYPEART_PLUGIN) $(STACK_ONLY_FLAGS) | llc $(TO_OBJECT_FILE)
# Link:
$> clang++ $(LINK_FLAGS) -L$(TYPEART_LIBPATH) -ltypeart-rt code.o -o binary
```

#### 1.1.2 Options for TypeART passes

The main options are shown below.

| Flag | Default | Description |
| --- | :---: | --- |
| `typeart` | - | Invoke typeart pass through LLVM `opt` |
| `typeart-outfile` | `types.yaml` | Serialized type layout information of user-defined types |
| `typeart-no-heap` | false | Do **not** instrument heap allocations |
| `typeart-alloca` | false | Instrument stack and global allocations |
| `typeart-stats` | false | Show instrumentation stat counters |
| `call-filter` | false | Filter stack and global allocations |
| `call-filter-str` | `*MPI_*` | Filter string target (glob string) |

##### Example invocations

###### Pre-requisites

1. Loading TypeARTS plugins with `opt`:
    ```shell
    TYPEART_PLUGIN=-load $(PLUGIN_PATH)/meminstfinderpass.so \
                   -load $(PLUGIN_PATH)/typeartpass.so`
    ```
2. Input of `opt` is LLVM IR, e.g.:
    ```shell
    # Pipe LLVM IR to console
    clang++ -g -Xclang -disable-llvm-passes -S -emit-llvm -o - example.cpp
    ```

###### Examples

- Invoke TypeART for heap-only instrumentation (with stats):
    ```shell
    opt $(TYPEART_PLUGIN) -typeart -typeart-stats
    ```
- Invoke TypeART for stack- and global-only instrumentation (*no* stats):
    ```shell
    opt $(TYPEART_PLUGIN) -typeart -typeart-no-heap=true -typeart-alloca
    ```
- Invoke TypeART for stack- and global-only instrumentation (with filtering):
    ```shell
    // Filter targets MPI by default:
    opt $(TYPEART_PLUGIN) -typeart -typeart-no-heap=true -typeart-alloca -call-filter
    // Filter target non-MPI API:
    opt $(TYPEART_PLUGIN) -typeart -typeart-no-heap=true -typeart-alloca -call-filter -call-filter-str=MY_API*
    ```
- Invoke TypeART for combined instrumentation (with filtering):
    ```shell
    opt $(TYPEART_PLUGIN) -typeart -typeart-alloca -call-filter
    ```

Also consult the [demo Makefile](demo/Makefile) for an example recipe, and flags for TypeART.

#### 1.1.3 Serialized type information (`types.yaml`) - Type-id

Static type information are serialized during instrumentation to a file `types.yaml`. Each user-defined type is
extracted and an integer `type-id` is attached to it. Built-ins (`float` etc.) have pre-defined `type-ids` and byte
layouts.

The TypeART instrumentation callbacks use the `type-id`. The runtime library correlates the allocation with the
respective type (and layout) during execution. Consider the following struct:

```c
struct s1_t {
  char a[3];
  struct s1_t* b;
}
```

The TypeART pass will write a `types.yaml` file with the following content:
<!--- @formatter:off --->
```yaml
- id: 256            // struct type-id
  name: struct.s1_t
  extent: 16
  member_count: 2
  offsets: [ 0, 8 ]  // byte offsets
  types: [ 0, 10 ]   // member type-ids (0-char, 10-ptr)
  sizes: [ 3, 1 ]    // array lengths
```
<!--- @formatter:on --->

#### 1.1.4 Filtering allocations (stack and global)

To improve performance, a translation unit-local (TU) data-flow filter for global and stack variables exist, see Section
1.1.2 *Options*. It follows the LLVM IR `use-def` chain. If the allocation provably never reaches the target API, it can
be filtered. Otherwise, it is instrumented.

Consider the following example.

1. The filter can remove `a`, as the aliasing pointer `x` is never part of an MPI call.
2. `b` is instrumented as the aliasing pointer `y` is part of an MPI call.
3. `c` is instrumented as we cannot reason about the body of `foo_bar`.

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

### 1.2 Executing an instrumented target code

### 1.3 Example: MPI Demo

The folder [demo](demo) contains an example of MPI related type errors that can be detected using TypeART. The code is
compiled with our instrumentation, and executed by preloading the MPI related check library implemented
in [tool.c](demo/tool.c), which is linked against the TypeART runtime and uses the aforementioned query interface. It
overloads the required MPI calls and checks that the passed `void*` buffer is correct.

## 2. Building TypeART

TypeART requires [LLVM](https://llvm.org) version 10 and CMake version >= 3.14.

### 2.1 Optional software requirements

- MPI library: Needed for some tests, the [demo](demo), our [MPI interceptor library](lib/mpi_interceptor), and for
  logging with our TypeART runtime library within an MPI target application.
- OpenMP-enabled Clang compiler: Needed for some tests.

Other smaller, external dependencies are defined within the [externals folder](externals) (depending on configuration
options), see Section 2.2.1 (Runtime). These are automatically downloaded during configuration time (internet required).

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

#### 2.2.1 CMake Configuration: Options for users

##### Runtime

<!--- @formatter:off --->
| Option | Default | Description |
| --- | :---: | --- |
| `USE_ABSL` | ON | Enable usage of btree-backed map of the [Abseil project](https://abseil.io/) instead of `std::map` |
| `USE_BTREE` | OFF | *Deprecated*. Enable usage of a [btree-backed map](https://github.com/ahueck/cpp-btree) (alternative to Abseil) instead of `std::map` |
| `SOFTCOUNTERS` | OFF | Enable runtime tracking of #tracked addrs. / #distinct checks / etc. |
| `LOG_LEVEL_RT` | 0 | Granularity of runtime logger. 3 ist most verbose, 0 is least |
<!--- @formatter:on --->

###### Runtime Thread-safety options

Default mode is to protect the global data structure with a (shared) mutex. Two main options exist:

<!--- @formatter:off --->
| Option | Default | Description |
| --- | :---: | --- |
| `DISABLE_THREAD_SAFETY` | OFF | Disable thread safety of runtime |
| `ENABLE_SAFEPTR` | OFF | Instead of a mutex, use a special data structure wrapper for concurrency, see [object_threadsafe](https://github.com/AlexeyAB/object_threadsafe) |
<!--- @formatter:on --->

##### LLVM Passes

<!--- @formatter:off --->
| Option | Default | Description |
| --- | :---: | --- |
| `SHOW_STATS` | ON | Passes show compile-time summary w.r.t. allocations counts |
| `MPI_INTERCEPT_LIB` | ON | Library to intercept MPI calls by preloading and check whether TypeART tracks the buffer pointer |
| `MPI_LOGGER` | ON | Enable better logging support in MPI execution context |
| `LOG_LEVEL` | 0 | Granularity of pass logger. 3 ist most verbose, 0 is least |
<!--- @formatter:on --->

##### Testing

<!--- @formatter:off --->
| Option | Default | Description |
| --- | :---: | --- |
| `TEST_CONFIG` | OFF | Set (force) logging levels to appropriate levels for test runner to succeed |
| `ENABLE_CODE_COVERAGE` | OFF | Enable code coverage statistics using LCOV 1.14 and genhtml (gcovr optional) |
| `ENABLE_LLVM_CODE_COVERAGE` | OFF | Enable llvm-cov code coverage statistics (llvm-cov and llvm-profdata  required) |
| `ENABLE_ASAN, TSAN, UBSAN` | OFF | Enable Clang sanitizers (tsan is mutually exlusive w.r.t. ubsan and  asan as they don't play well together) |
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
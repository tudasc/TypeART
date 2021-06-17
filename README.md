# TypeART &middot; [![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) ![](https://github.com/tudasc/TypeART/workflows/TypeART-CI/badge.svg?branch=master) ![](https://github.com/tudasc/TypeART/workflows/TypeART-CI-ext/badge.svg?branch=master) [![Coverage Status](https://coveralls.io/repos/github/tudasc/TypeART/badge.svg?branch=master)](https://coveralls.io/github/tudasc/TypeART)

## What is TypeART?

TypeART \[[TA18](#ref-typeart-2018); [TA20](#ref-typeart-2020)\] is a type and memory allocation tracking sanitizer. It
consists of an LLVM compiler pass plugin and a corresponding runtime to track memory allocations during the execution of
a target program. It instruments heap, stack and global variable allocations with a callback to our runtime. The
callback consists of 1) the memory address, 2) the type-layout of the allocation (built-ins, user-defined structs etc.)
and 3)
extent of the value. This allows users of our runtime to query detailed type information behind arbitrary memory
locations, as long as they are mapped. TypeART also works with OpenMP codes and its runtime is thread-safe (since
release 1.6).

## Why use it?

Employ TypeART whenever you need type information of allocations in your program to verify some property, and generate
diagnostics if it doesn't hold. For instance, low-level C-language APIs use `void`-pointers as generic types to call
some library function. The user must specify its type and length manually. With TypeART, it is straightforward to verify
that a `void`-pointer argument to an API is, e.g., a type `T` array with length `n`. Examples for type unsafe APIs
include the Message-Passing Interface (MPI), checkpointing libraries and numeric solver libraries.

### Use Case: MUST - A dynamic MPI correctness checker

MUST \[[MU13](#ref-must-2013)\] is a dynamic MPI correctness checker that is able to, e.g., detect deadlocks or a
mismatch of MPI datatypes of the sending and receiving process, see
its [project page](https://www.hpc.rwth-aachen.de/must/). MUST relies on intercepting MPI calls for its analysis. As a
consequence, though, MUST is unaware of the *effective*
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
download the current release of MUST with TypeART (1.8 or higher) on
the [MUST homepage](https://itc.rwth-aachen.de/must/).

### References

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

## Using TypeART

Making use of TypeART consists of two phases:

1. Compile your code with Clang/LLVM-10 using the TypeART LLVM pass plugins to 1) extract static type information to a
   `yaml`-file and 2) instrument all relevant allocations.
2. Execute the target program with a runtime library (a **client** based on the TypeART runtime) to accept the callbacks
   from the instrumented code and actually do some useful analysis. To that end, the
   interface [RuntimeInterface.h](lib/runtime/RuntimeInterface.h) can be used as a type-information query interface for
   clients.

### 1. Compiling a target code

Our LLVM compiler pass plugins instrument allocations and also serialize the static type layouts of these allocations to
a yaml file (default name `types.yaml`).

#### Building with TypeART

A typical compile invocation may first compile code to object files and then link with any libraries, e.g.:

```shell
# Compile with Clang
$> clang -O2 $COMPILE_FLAGS -c code.cpp -o code.o
# Link
$> clang $LINK_FLAGS code.o -o binary
```

With TypeART this recipe needs to be changed, as we rely on the LLVM `opt` (optimizer) tool to load and apply our
TypeART passes to a target code based on the LLVM intermediate representation. To that end, the following steps are
currently required:

1. Compile the code down to LLVM IR, and pipe the output to the LLVM `opt` tool. (Keeping your original compile flags)
2. Apply heap instrumentation with TypeART.
3. Optimize the code with -Ox using `opt`.
4. Apply stack and global instrumentation with TypeART.
5. Pipe the final output to LLVM `llc` to generate the final object file

Once the object file is created, the link step needs to link the TypeART runtime (for the added instrumentation
callbacks).

```shell
# Compile
$> clang $COMPILE_FLAGS $EMIT_LLVM_IR_FLAGS code.cpp | opt $TYPEART_PLUGIN $HEAP_ONLY_FLAGS | opt -O2 | opt $TYPEART_PLUGIN $STACK_ONLY_FLAGS | llc $TO_OBJECT_FILE  
# Link
$> $CXX $LINK_FLAGS -L$(TYPEART_LIBPATH) -ltypeart-rt  code.o -o binary
```

##### LLVM compiler pass

The analysis pass finds all heap, stack and global allocations. Based on a data-flow filter, it discards stack and
global allocation if they are never passed to a specified API (default: `MPI`). Subsequently, it serializes the
user-defined types (`struct`, `class` etc.) of each allocation and attaches a so-called type-id to each type for the
runtime callbacks. The instrumentation pass uses the filtered set of allocations and adds instrumentation callbacks to
our runtime, passing 1) the memory pointer, 2) the corresponding type-id and 3) the number of allocated elements (
extent). See below for an example instrumentation.

###### Example instrumentation in LLVM intermediate representation (IR)

- Original LLVM IR code, a simple heap allocation with malloc of a `double`-array. It contains all information needed
  for TypeART:

    ~~~llvm
    ; Assume: %size == n * sizeof(double)
    %pointer = call i8* @malloc(i64 %size)
    %pointer_double = bitcast i8* %pointer to double*
    ~~~

- Corresponding TypeART instrumentation:

    ~~~llvm
    ; Assume: %size == n * sizeof(double)
    ; Heap allocation -- %pointer holds the returned pointer value:
    %pointer = call i8* @malloc(i64 %size)
    ; TypeART instrumentation -- compute the number of double-elements:
    %extent = udiv i64 %size, 8
    ; TypeART runtime callback (pointer, type-id, length) -- 6 is the type-id for double:
    call void @__typeart_alloc(i8* %pointer, i32 6, i64 %extent)
    ; Original bitcast:
    %pointer_double = bitcast i8* %pointer to double*
    ~~~

###### Type-id (`types.yaml`)

### 2. Executing an instrumented target code

### Example: MPI Demo

The folder [demo](demo) contains an example of MPI related type errors that can be detected using TypeART. The code is
compiled with our instrumentation, and executed by preloading the MPI related check library implemented
in [tool.c](demo/tool.c), which is linked against the TypeART runtime and uses the aforementioned query interface. It
overloads the required MPI calls and checks that the passed `void* buffer` is correct.

## Building TypeART

TypeART requires [LLVM](https://llvm.org) version 10 and CMake version >= 3.14.

### Building

TypeART uses CMake to build, cf. [GitHub CI build file](.github/workflows/basic-ci.yml) for a complete recipe to build.
Example build recipe (debug build, installs to default prefix
`${typeart_SOURCE_DIR}/install/typeart`)

```sh
$> git clone https://github.com/tudasc/TypeART
$> cd TypeART
$> cmake -B build
$> cmake --build build --target install --parallel
```

#### CMake Configuration: Options for users

##### Runtime

- `SOFTCOUNTERS` (default: **off**) : Enable runtime tracking of #tracked addrs. / #distinct checks / etc.
- `USE_ABSL` (default: **on**) : Enable usage of btree-backed map of the [Abseil project](https://abseil.io/) instead of
  std::map for the runtime.
- `USE_BTREE` (default: **off**) : *Deprecated* Enable usage of
  a [btree-backed map](https://github.com/ahueck/cpp-btree) (alternative to Abseil) instead of std::map for the runtime.

###### Runtime Thread safety options

Default mode is to protect the global data structure with a (shared) mutex. Two main options exist:

- `DISABLE_THREAD_SAFETY` (default: **off**) : Disable thread safety of runtime
- `ENABLE_SAFEPTR` (default: **off**) : Instead of a mutex, use a special data structure wrapper for concurrency,
  see [object_threadsafe](https://github.com/AlexeyAB/object_threadsafe)

##### Logging and Passes

- `SHOW_STATS` (default: **on**) : Passes show compile-time summary w.r.t. allocations counts.
- `MPI_LOGGER` (default: **on**) : Enable better logging support in MPI execution context
- `MPI_INTERCEPT_LIB` (default: **on**) : Library can be used by preloading to intercept MPI calls and check whether
  TypeART tracks the buffer pointer
- `LOG_LEVEL_` and `LOG_LEVEL_RT` (default **0**) :  Granularity of logger. 3 ist most verbose, 0 is least.

##### Testing

- `TEST_CONFIG` (default: **off**) : Set (force) logging levels to appropriate levels for test runner to succeed
- `ENABLE_CODE_COVERAGE` (default: **off**) : Enable code coverage statistics using LCOV 1.14 and genhtml (gcovr
  optional)
- `ENABLE_LLVM_CODE_COVERAGE` (default: **off**) : Enable llvm-cov code coverage statistics (llvm-cov and llvm-profdata
  required)
- `ENABLE_ASAN, TSAN, UBSAN` (default: **off**) : Enable Clang sanitizers (tsan is mutually exlusive w.r.t. ubsan and
  asan as they don't play well together)


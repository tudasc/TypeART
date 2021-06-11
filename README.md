# TypeART &middot; [![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) ![](https://github.com/tudasc/TypeART/workflows/TypeART-CI/badge.svg?branch=master) ![](https://github.com/tudasc/TypeART/workflows/TypeART-CI-ext/badge.svg?branch=master) [![Coverage Status](https://coveralls.io/repos/github/tudasc/TypeART/badge.svg?branch=master)](https://coveralls.io/github/tudasc/TypeART)

TypeART \[[TA18](#ref-typeart-2018); [TA20](#ref-typeart-2020)\] is a type and memory allocation tracking sanitizer. It
consists of an LLVM compiler pass and a corresponding runtime to track relevant memory allocation information during the
execution of a target program. It instruments heap, stack and global variable allocations with a callback to our
runtime. The callback consists of the runtime memory pointer value, what type (built-ins, user-defined structs etc.) and
extent of the value. This allows users of our runtime to query detailed type information behind arbitrary memory
locations, as long as they are mapped. TypeART works with OpenMP codes and its runtime is thread-safe (since release
1.6).

### Use Case: MUST - A dynamic MPI correctness checker

MUST \[[MU13](#ref-must-2013)\] is a dynamic MPI correctness checker that is able to, e.g., detect deadlocks or a
mismatch of MPI datatypes of the sending and receiving process, see
its [project page](https://www.hpc.rwth-aachen.de/must/). MUST relies on intercepting MPI calls for its analysis. As a
consequence, though, MUST is unaware of the *effective*
type of the allocated `void*` buffers used for the low-level MPI API. To that end, TypeART was developed to track
memory (de-)allocation relevant to MPI communication. With TypeART, MUST can check for type compatibility between the
type-less communication buffer and the declared MPI datatype at all phases of the MPI communication, namely message
assembly, message transfer and message disassembly into the receiving buffer.

#### Type checking for MPI calls

Consider the MPI function `MPI_Send(const void* buffer, int count, MPI_Datatype datatype, ...)`. Without TypeART, MUST
cannot check 1) if the `buffer` argument is compatible with the declared `MPI_Dataype` and 2) if its minimum size
is `count`. For instance, if the datatype is `MPI_DOUBLE`, we expect the `buffer` argument to be a `double*`-array with
a minimum size of `count`:

```{.c}
// TypeART tracks this allocation: memory address, type and size
double* array = (double*) malloc(length*sizeof(double));
// MUST intercepts this MPI call, and asks TypeARTs runtime library for type information:
//   1. Is the first argument of type double?
//   2. Is the allocation at least of size *length*? 
MPI_Send((void*) array, length, MPI_Double, ...)
```

MUST and TypeART also handle MPI [derived datatypes](https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report/node77.htm)
with complex underlying datastructures. For more details, see our publications below or download the current release of
MUST with TypeART (1.8 or higher) on the [MUST homepage](https://itc.rwth-aachen.de/must/).

#### References

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

1. Compile and instrument the target code with our LLVM passes, and,
2. execute the target program with a runtime library (based on the TypeART runtime) to accept the callbacks from the
   instrumented code and actually do some useful analysis.

To that end, the interface [RuntimeInterface.h](runtime/RuntimeInterface.h) can be used to query type information during
the target code execution.

### 1. Compiling a target code

#### LLVM pass

The LLVM pass instruments memory allocations, and extracts their type information. TypeART handles heap, stack and
global variables. An optional static data-flow filter discards (during compilation) stack and global variables if, e.g.,
they are never passed to an MPI call. For each relevant allocation, the pass determines the 1) memory address, 2)
extent/size of the allocation and 3) a type-id is generated for the instumentation.

###### Type-id

The type information is necessary to correlate the type of the buffer passed to an MPI call with the MPI datatype the
user declared.

##### Example of Instrumentation: Handling malloc

To instrument relevant allocations and extract the necessary type information, the LLVM pass searches for specific
patterns, e.g., how calls to ```malloc``` look like in LLVM IR. Calls to the ```malloc``` function are typically call
instructions followed by a ```bitcast``` instruction to cast the returned pointer to the desired type.

~~~{.ll}
; %0 == n * sizeof(float)
%1 = tail call i8* @malloc(i64 %0)
%2 = bitcast i8* %1 to float*
~~~

The patterns has all the information we require for our instrumentation. Our transformation first detects the type that
the returned pointer is casted to, then it computes the extent of the allocation. The information is passed to our
instrumentation function.

~~~{.ll}
; %0 == n * sizeof(float)
; %1 holds the returned pointer value
%1 = tail call i8* @malloc(i64 %0)
; compute the number of elements
%2 = udiv i64 %0, 4
; call TypeART runtime (5 is the type-id for float)
call void @__typeart_alloc(i8 *%1, i32 5, i64 %2)
; original bitcast
%3 = bitcast i8* %1 to float *
~~~

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

```{.sh}
$> git clone https://github.com/tudasc/TypeART
$> cd TypeART
$> cmake -B build
$> cmake --build build --target install --parallel
```

#### CMake Configuration: Options for users

##### Runtime

- `SOFTCOUNTERS` (default: **off**) : Enable runtime tracking of #tracked addrs. / #distinct checks / etc.
- `USE_ABSL` (default: **on**) : Enable usage of btree-backed map of the abseil project instead of std::map for the
  runtime.
- `USE_BTREE` (default: **off**) : *Deprecated* Enable usage of a btree-backed map (alternative to abseil) instead of
  std::map for the runtime.

###### Runtime Thread safety options

Default mode is to protect the global data structure with a (shared) mutex. Two main options exist:

- `DISABLE_THREAD_SAFETY` (default: **off**) : Disable thread safety of runtime
- `ENABLE_SAFEPTR` (default: **off**) : Instead of a mutex, use a special data structure wrapper for concurrency,
  see [object_threadsafe](https://github.com/AlexeyAB/object_threadsafe)

##### Logging and Passes

- `SHOW_STATS` (default: **on**) : Passes show the statistics w.r.t. allocations etc.
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


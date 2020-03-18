# TypeART

TypeART \[[TA18](#ref-typeart-2018)\] is a type and memory allocation tracking sanitizer.
It consists of an LLVM compiler pass and a corresponding runtime to track relevant memory allocation information during the execution of a target program.
It instruments heap, stack and global variable allocations with a callback to our runtime. 
The callback consists of the runtime memory pointer value, what type (built-ins, user-defined structs etc.) and extent of the value.
This allows users of our runtime to query detailed type information behind arbritary memory locations, as long as they are mapped.

### Use Case: MUST - A dynamic MPI correctness checker

TypeART is used in conjunction with MUST \[[MU13](#ref-must-2013)\] to track memory (de-)allocation relevant to MPI communication.
Thus, MUST can check for type compatibility between the type-less communication buffer and the declared MPI datatype at all phases of the MPI communication, namely message assembly, message transfer and message disassembly into the receiving buffer.
A brief summary is given in a subsequent section and more information can be found in our publication:

#### References

<table style="border:0px">
<tr>
    <td valign="top"><a name="ref-typeart-2018"></a>[TA18]</td>
    <td>Hück, Alexander and Lehr, Jan-Patrick and Kreutzer, Sebastian and Protze, Joachim and Terboven, Christian and Bischof, Christian and Müller, Matthias S.
    <a href=http://conferences.computer.org/scw/2018/pdfs/Correctness2018-4a8nikwzUlkPjw1TP5zWZt/3eQuPpEOKXTkjmMgQI3L3T/5g7rbAUBoYPUZJ6duKhpL4.pdf>
    Compiler-aided type tracking for correctness checking of MPI applications</a>.
    In <i>2nd International Workshop on Software Correctness for HPC Applications (Correctness)</i>,
    pages 51–58. IEEE, 2018.</td>
</tr>
<tr>
    <td valign="top"><a name="ref-must-2013"></a>[MU13]</td>
    <td>Hilbrich, Tobias and Protze, Joachim and Schulz, Martin and de Supinski, Bronis R. and Müller, Matthias S.
    <a href=http://dx.doi.org/10.3233/SPR-130368>
    MPI Runtime Error Detection with MUST: Advances in Deadlock Detection</a>.
    In <i>Scientific Programming</i>, vol. 21, no. 3-4,
    pages 109–121, 2013.</td>
</tr>
</table>

## Software dependencies

TypeART requires [LLVM](https://llvm.org) version 6.0 and CMake version >= 3.5.

#### Building TypeART

TypeART uses CMake to build, cf. [TravisCI build file](.travis.yaml) for a complete recipe to build.
```{.sh}
$> git clone https://github.com/jplehr/TypeART.git
$> cd TypeART
$> mkdir build && cd build
$> cmake .. -DCMAKE_INSTALL_PREFIX=*your path*
$> cmake --build . --target install
```

#### CMake Configuration: Options for users

- `SOFTCOUNTERS` (default: **off**) : Enable runtime tracking of #tracked addrs. / #distinct checks / etc.
- `USE_BTREE` (default: **on**) : Enable usage of btree-backed map instead of std::map for the runtime, typically resulting in higher performance. 

## Using TypeART

Making use of TypeART consists of two phases: 

  1. Compile and instrument the target code with our LLVM passes, and, 
  2. execute the target program with a runtime library (based on the TypeART runtime) to accept the callbacks from the instrumented code and actually do some useful analysis.

To that end, the interface [RuntimeInterface.h](runtime/RuntimeInterface.h) can be used to query type information during the target code execution.

#### Example: MPI Demo

The folder [demo](demo) contains an example of MPI related type errors that can be detected using TypeART.
The code is compiled with our instrumentation, and executed by preloading the MPI related check library implemented in [tool.c](demo/tool.c), which is linked against the TypeART runtime and uses the aforementioned query interface.
It overloads the required MPI calls and checks that the passed `void* buffer` is correct.

## LLVM pass

The necessary allocation sites and type information are extracted in LLVM passes.
TypeART analyzes:

- Calls to ```malloc``` and ```free``` to keep track of active pointers referring to objects allocated on the heap,
- relevant stack space allocations, i.e.,  allocations that cannot be proven to never lead to ```MPI``` functions,
- built-in as well as user-defined types to retrieve type size and the size of the allocation, e.g., for arrays.

The type information is necessary to correlate the type of the buffer passed to an MPI call with the MPI datatype the user declared.
In this prototype we restrict ourselves to:

+ primitive types (int, float, long, double, char, unsigned int, unsigned long)
+ arrays of primitive types
+ structs which contain only primitive types

### Example of Instrumentation: Handling malloc

To instrument relevant allocations and extract the necessary type information, the LLVM pass searches for specific patterns, e.g., how calls to ```malloc``` look like in LLVM IR.
Calls to the ```malloc``` function are typically call instructions followed by a ```bitcast``` instruction to cast the returned pointer to the desired type.

~~~{.ll}
; %0 == n * sizeof(float)
%1 = tail call i8* @malloc(i64 %0)
%2 = bitcast i8* %1 to float *
~~~

The patterns has all the information we require for our instrumentation.
Our transformation first detects the type that the returned pointer is casted to, then it computes the extent of the allocation.
The information is passed to our instrumentation function.

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

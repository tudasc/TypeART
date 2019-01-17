# TypeART

TypeART is a type and memory allocation tracking sanitizer.
It consists of a LLVM compiler pass and a corresponding runtime to track relevant memory allocation information during the execution of a target program.
It instruments heap, stack and global variable allocations with a callback to our runtime. 
The callback consists of the runtime memory pointer value, what type (built-ins, user-defined structs etc.) and extent of the value.
This allows users of our runtime to query detailed type information behind arbritary memory locations, as long as they are mapped.

### Use Case: MUST - A dynamic MPI correctness checker
TypeART is used in conjunction with [MUST](https://doc.itc.rwth-aachen.de/display/CCP/Project+MUST) to track memory (de-)allocation relevant to MPI communication.
Thus, MUST can check for type compatibility between the type-less communication buffer and the declared MPI datatype at all phases of the MPI communication, namely message assembly, message transfer and message disassembly into the receiving buffer.
A brief summary is given in a subsequent section and more information can be found in our publication:
[Compiler-aided type tracking for correctness checking of MPI applications](http://conferences.computer.org/scw/2018/pdfs/Correctness2018-4a8nikwzUlkPjw1TP5zWZt/3eQuPpEOKXTkjmMgQI3L3T/5g7rbAUBoYPUZJ6duKhpL4.pdf).

## Software dependencies
TypeART requires [LLVM](https://llvm.org) version 6.0 and CMake version >= 3.5.

#### Building TypeART
TypeART uses CMake to build, cf. [TravisCI build file](.travis.yaml) for a complete recipe to build.
```{.sh}
$> git clone https://github.com/jplehr/TypeART.git
$> cd TypeART
$> mkdir build && cd build
$> cmake ..
$> cmake --build .
```

## Using TypeART
TypeART can be applied to a source file by invoking the script ```applyPass.sh```:
```{.sh}
$> scripts/applyPass.sh target.cpp /path/to/plugin
```
The tool uses ```/tmp``` to store temporary files.


## LLVM pass
The necessary allocation sites and type information are extracted in LLVM passes.
TypeART analyzes:
- Calls to ```malloc``` and ```free``` to keep track of active pointers referring to objects allocated on the heap,
- relevant stack space allocations, i.e.,  allocations that cannot be proven to never lead to ```MPI``` functions,
- built-in as well as user-defined types to retrieve type size and the size of the allocation, e.g., for arrays,

The type information is necessary to correlate the type of the buffer passed to an MPI call with the MPI datatype the user declared.
In this prototype we restrict ourselves to:
+ primitive types (int, float, long, double, char, unsigned int, unsigned long)
+ arrays of primitive types
+ structs which contain only primitive types

### malloc
Calls to the ```malloc``` function are typically call instructions followed by a ```bitcast``` instruction to cast the pointer to the desired type.

~~~{.ll}
%1 = tail call i8* @malloc(i64 168)
%2 = bitcast i8* %1 to i32*
~~~

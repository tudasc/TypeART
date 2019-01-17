# TypeART

TypeART is a compiler plugin + runtime to gather and track type information for every allocation site relevant to Message Passing Interface (MPI) API function calls.
Together with the [MUST dynamic MPI checker](https://doc.itc.rwth-aachen.de/display/CCP/Project+MUST) this enables a user to check the correct construction and usage of MPI built-in, as well as user-defined types.
A brief summary is given in a subsequent section and more information can be found [in our paper](https://sc18.supercomputing.org/proceedings/workshops/workshop_pages/ws_corr102.html).

## Software dependencies
TypeART requires [LLVM](https://llvm.org) in version 6.0 and CMake >= 3.5.
The runtime optionally uses Google's BTree implementation that is downloaded during the build process.

## Building TypeART
TypeART uses CMake to build.
```{.sh}
$> git clone https://github.com/jplehr/TypeART.git
$> cd TypeART
$> mkdir build
$> cmake ..
$> make -j
$> make install
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


### structs
As far as I see it, we would only be interested in struct **definitions** not declarations, as we cannot generate useful information from declarations on their own.


## Runtime
- maintain map of neccessary information
  + I think currently that's: [ptr_value, type_encoding, count, type_elem_size]
- provide access function taking a pointer and returning a type encoding, e.g. *must_type_enc_t getType(std::intptr_t p)*

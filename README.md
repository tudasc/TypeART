# llvm-must-support

Repository for the prototype implementation of an LLVM pass plus runtime, which implements type tracking to help MUST validate the correct use of MPI calls.

## LLVM pass
- malloc/free instrumentation to keep track of active pointers
- type information retrieval
- type information encoding in MUST internal representation (?)

The type information is necessary to correlate the type of the buffer passed to an MPI call with the MPI datatype the user declared.
In this prototype we restrict ourselves to:
+ primitive types (int, float, long, double, char, unsigned int, unsigned long)
+ arrays of primitive types
+ structs which contain only primitive types


### malloc
Calls to the malloc function are typically call instructions followed by a bitcast instruction to cast the pointer to the desired type.

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

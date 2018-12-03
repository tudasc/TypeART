# TypeART

TypeART is a compiler plugin + runtime to gather and track type information for every allocation site relevant to Message Passing Interface (MPI) API function calls.
Together with the MUST dynamic MPI checker this enables a user to check the correct construction and usage of MPI built-in,  as well as user-defined types.
More information can be found [here](https://sc18.supercomputing.org/proceedings/workshops/workshop_pages/ws_corr102.html).

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

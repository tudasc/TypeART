# llvm-must-support

Repository for the prototype implementation of an LLVM pass plus runtime, which implements type tracking to help MUST validate the correct use of MPI calls.

## LLVM pass
- malloc/free instrumentation to keep track of active pointers
- type information retrieval
- type information encoding in MUST internal representation (?)

## Runtime
- maintain map of neccessary information
  + I think currently that's: [ptr_value, type_encoding, count, type_elem_size]
- provide access function taking a pointer and returning a type encoding, e.g. *must_type_enc_t getType(std::intptr_t p)*

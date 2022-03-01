# MPI Interceptor Library

The MPI Interceptor Library handles the type checking of MPI calls using the TypeART Runtime System. It injects itself
between the caller and the callee of every MPI send and/or receive function and logs information about the types and
buffer sizes used as arguments to each call and warns when the buffer type does not match the MPI type or the size of
any buffer given as an argument is insufficient w.r.t. the arguments of the call.

## How to use this Documentation

This document gives an overview of whether and how any given MPI function is type checked. If for any MPI call TypeART
prints an error which you would not expect, please first refer to this document to check whether your usecase is covered
by this library. If you think that you found a reasonable usecase that should be type checked successfully but is not
currently covered by this library please create an issue in
the [TypeART GitHub repository](https://github.com/tudasc/TypeART/issues).

## Unsupported Send/Receive Functions

The following function are currently not type checked:

- [MPI_Alltoallw](https://www.open-mpi.org/doc/v4.1/man3/MPI_Alltoallw.3.php)
- [MPI_Ialltoallw](https://www.open-mpi.org/doc/v4.1/man3/MPI_Ialltoallw.3.php)
- [MPI_Ineighbor_alltoallw](https://www.open-mpi.org/doc/v4.1/man3/MPI_Ineighbor_alltoallw.3.php)
- [MPI_Neighbor_alltoallw](https://www.open-mpi.org/doc/v4.1/man3/MPI_Neighbor_alltoallw.3.php)

## Custom MPI Type Support

MPI provides a number of type combinators which can create new user-defined types based on existing MPI types. A list of
all type combiners can be found
[here](https://www.open-mpi.org/doc/v4.1/man3/MPI_Type_get_envelope.3.php#toc8).

### Unsupported Type Combiners

- [MPI_Type_hindexed](https://www.open-mpi.org/doc/v4.1/man3/MPI_Type_hindexed.3.php)
- [MPI_Type_create_hindexed_block](https://www.open-mpi.org/doc/v4.1/man3/MPI_Type_create_indexed_block.3.php)
- [MPI_Type_create_darray](https://www.open-mpi.org/doc/v4.1/man3/MPI_Type_create_darray.3.php)
- [MPI_Type_create_f90_real](https://www.open-mpi.org/doc/v4.1/man3/MPI_Type_create_f90_real.3.php)
- [MPI_Type_create_f90_complex](https://www.open-mpi.org/doc/v4.1/man3/MPI_Type_create_f90_complex.3.php)
- [MPI_Type_create_f90_integer](https://www.open-mpi.org/doc/v4.1/man3/MPI_Type_create_f90_integer.3.php)
- [MPI_Type_create_resized](https://www.open-mpi.org/doc/v4.1/man3/MPI_Type_create_resized.3.php)

### Handling of ambiguous types

In cases where a variable has a struct type where the first member of that type has an offset of 0 bytes, the runtime
cannot discern whether an address to an instance of this struct is meant to have the type of that struct or of the first
member. In these cases the runtime first tries to match the MPI type to the struct type and then, if that fails,
recursively attempts to match it to the type of the first member.

### [MPI_Type_dup](https://www.open-mpi.org/doc/v4.1/man3/MPI_Type_dup.3.php)

The original type will be used for the type check. No special limitations apply.

### [MPI_Type_contiguous](https://www.open-mpi.org/doc/v4.1/man3/MPI_Type_contiguous.3.php)

Matches any buffer that is an array and can hold at least `count` elements of a datatype that matches `oldtype`.

### [MPI_Type_vector](https://www.open-mpi.org/doc/v4.1/man3/MPI_Type_vector.3.php)

Matches any buffer that is an array and can hold at least `(count - 1) * stride + blocklength`
elements of a datatype that matches `oldtype`.

Note: negative strides are currently unsupported.

### [MPI_Type_create_indexed_block](https://www.open-mpi.org/doc/v4.1/man3/MPI_Type_create_indexed_block.3.php)

Matches any buffer that is an array and can hold at least `max(array_of_displacements) + blocklength`
elements of a datatype that matches `oldtype`.

Note: negative displacements are currently unsupported.

### [MPI_Type_create_struct](https://www.open-mpi.org/doc/v4.1/man3/MPI_Type_create_struct.3.php)

Matches any buffer that has a struct type where

- the struct has the same number of members as the MPI type,
- the offset of each member (as determined by the
  [offsetof-operator](https://en.cppreference.com/w/cpp/types/offsetof))
  matches the displacement in the MPI type,
- the type of each member matches the respective MPI type and
- where the buffer can hold at least `count` instances of the struct type.

### [MPI_Type_create_subarray](https://www.open-mpi.org/doc/v4.1/man3/MPI_Type_create_subarray.3.php)

Matches any buffer that is an array and can hold at least as many elements as the product of all `array_of_sizes`
elements of a datatype that matches `oldtype`.

# MPI Interceptor Library

The MPI Interceptor Library handles the typechecking of MPI calls using the
TypeART Runtime System. It injects itself between the caller and the callee of
every MPI send and/or receive function and logs information about the types and
buffer sizes used as arguments to each call and warns when the buffer type does
not match the MPI type or the size of any buffer given as an argument is
insufficient w.r.t. the arguments of the call.

## How to use this Documentation

This document gives an overview of whether and how any given MPI function is
typechecked. If for any MPI call TypeART prints an error which you would not
expect, please first refer to this document to check whether your usecase is
covered by this library. If you think that you found a reasonable usecase that
should be typechecked successfully but is not currently covered by this library
please create an issue in the [TypeART GitHub repository](https://github.com/tudasc/TypeART/issues).

## Unsupported Send/Receive Functions

The following function are currently not typechecked:

- [MPI_Alltoallw](https://www.open-mpi.org/doc/v4.1/man3/MPI_Alltoallw.3.php)
- [MPI_Ialltoallw](https://www.open-mpi.org/doc/v4.1/man3/MPI_Ialltoallw.3.php)
- [MPI_Ineighbor_alltoallw](https://www.open-mpi.org/doc/v4.1/man3/MPI_Ineighbor_alltoallw.3.php)
- [MPI_Neighbor_alltoallw](https://www.open-mpi.org/doc/v4.1/man3/MPI_Neighbor_alltoallw.3.php)

## Custom MPI Type Support

MPI provides a number of type combinators which can create new user-defined
types based on existing MPI types. A list of all type combiners can be found
[here](https://www.open-mpi.org/doc/v4.1/man3/MPI_Type_get_envelope.3.php#toc8).

### Unsupported Type Combiners

- [MPI_Type_hvector](https://www.open-mpi.org/doc/v4.1/man3/MPI_Type_hvector.3.php)
- [MPI_Type_hindexed](https://www.open-mpi.org/doc/v4.1/man3/MPI_Type_hindexed.3.php)

### [MPI_Type_dup](https://www.open-mpi.org/doc/v4.1/man3/MPI_Type_dup.3.php)

The original type will be used for the typecheck. No special limitations apply.

### [MPI_Type_contiguous](https://www.open-mpi.org/doc/v4.1/man3/MPI_Type_contiguous.3.php)

Matches any buffer that is an array and can hold at least `count` elements of a
datatype that matches `oldtype`.

### [MPI_Type_vector](https://www.open-mpi.org/doc/v4.1/man3/MPI_Type_vector.3.php)

Matches any buffer that is an array and can hold at least `(count - 1) * stride + blocklength`
elements of a datatype that matches `oldtype`.

Note: negative strides are currently unsupported.

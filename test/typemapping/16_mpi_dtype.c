// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart --typeart-stack=true  -S | %filecheck %s
// RUN: %c-to-llvm %s | %apply-typeart --typeart-type-serialization=inline --typeart-stack=true -S | %filecheck %s --check-prefix inline
// RUN: %c-to-llvm %s | %apply-typeart --typeart-type-serialization=hybrid --typeart-stack=true -S | %filecheck %s --check-prefix hybrid

// REQUIRES: llvm-18 || llvm-19
// clang-format on
struct ompi_struct_data;
typedef struct ompi_struct_data* MPI_Datatype;

extern MPI_Datatype MPI_DOUBLE;
extern MPI_Datatype MPI_INT;

int main(void) {
  MPI_Datatype array_of_types[3] = {MPI_DOUBLE, MPI_DOUBLE, MPI_INT};

  return 0;
}

// CHECK-NOT: Error
// CHECK: call {{.*}} @__typeart_alloc_stack(ptr {{.*}}, i32 1, i64 3)
// inline: @_typeart_ptr = weak_odr global %struct._typeart_struct_layout_t
// inline: call {{.*}} @__typeart_alloc_stack_mty(ptr {{.*}}, ptr {{.*}}, i64 3)
// hybrid: call {{.*}} @__typeart_alloc_stack(ptr {{.*}}, i32 1, i64 3)

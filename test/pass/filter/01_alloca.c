// Template for alloca.ll.in
// RUN: %filecheck %s
// XFAIL: *

#include <stdlib.h>

extern void bar(int x);
extern void ebar(int* x);

void MPI_send(void* a) {
  (void)a;
  return;
}

void foo_bar(int* a) {
  *a = 1;
  (void)a;
  return;
}

void foo_bar2(int* a, int* b) {
  *a = 1;
  (void)a;
  MPI_send(b);
  return;
}

void foo_bar3() {
  int* a = malloc(10 * sizeof(int));
  (void)a;
  MPI_send(a);
  return;
}

void foo() {
  int a  = 1;
  int b  = a;
  int* c = &a;
  int d  = a;
  int x  = a;
  //  MPI_send(&a);
  MPI_send(c);  // mem2reg a gets filtered because c points to alloca of a; withpout mem2reg the c alloca gets filtered
  bar(d);
  ebar(&d);
  foo_bar(&a);       // filter a
  foo_bar2(&a, &x);  // no filter a
}

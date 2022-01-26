// RUN: echo --- > types.yaml
// RUN: %wrapper-cc %s -o %s.exe
// RUN: %s.exe 2>&1 | %filecheck %s

// RUN: %wrapper-cc %s
// RUN: %p/a.out 2>&1 | %filecheck %s

// RUN: %wrapper-cc -O1 %s
// RUN: %p/a.out 2>&1 | %filecheck %s --allow-empty

// RUN: %wrapper-cc -MD -MT %s.o -MF %s.o.d  %s
// RUN: %p/a.out 2>&1 | %filecheck %s

int main(int argc, char** argv) {
  int a = 0;
  return a;
}

// CHECK-NOT: Error

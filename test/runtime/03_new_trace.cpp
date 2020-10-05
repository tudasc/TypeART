// RUN: %scriptpath/applyAndRun.sh %s %pluginpath "-typeart-alloca" %rtpath 2>&1 | FileCheck %s

template <typename T>
void new_delete() {
  T* t = new T;
  delete t;
}

template <typename T>
void new_delete(int n) {
  T* t = new T[n];
  delete[] t;
}

int main(int argc, char** argv) {
  const int n = 42;

  // CHECK: [Trace] TypeART Runtime Trace

  // CHECK: [Trace] Alloc 0x{{.*}} int8 1 1
  // CHECK: [Trace] Free 0x{{.*}}
  new_delete<char>();

  // CHECK: [Trace] Alloc 0x{{.*}} int16 2  1
  // CHECK: [Trace] Free 0x{{.*}}
  new_delete<short>();

  // CHECK: [Trace] Alloc 0x{{.*}} int32 4 1
  // CHECK: [Trace] Free 0x{{.*}}
  new_delete<int>();

  // CHECK: [Trace] Alloc 0x{{.*}} int64 8 1
  // CHECK: [Trace] Free 0x{{.*}}
  new_delete<long>();

  // CHECK: [Trace] Alloc 0x{{.*}} float 4 1
  // CHECK: [Trace] Free 0x{{.*}}
  new_delete<float>();

  // CHECK: [Trace] Alloc 0x{{.*}} double 8 1
  // CHECK: [Trace] Free 0x{{.*}}
  new_delete<double>();

  // CHECK: [Trace] Alloc 0x{{.*}} pointer 8 1
  // CHECK: [Trace] Free 0x{{.*}}
  new_delete<int*>();

  // CHECK: [Trace] Alloc 0x{{.*}} int8 1 42
  // CHECK: [Trace] Free 0x{{.*}}
  new_delete<char>(n);

  // CHECK: [Trace] Alloc 0x{{.*}} int16 2 42
  // CHECK: [Trace] Free 0x{{.*}}
  new_delete<short>(n);

  // CHECK: [Trace] Alloc 0x{{.*}} int32 4 42
  // CHECK: [Trace] Free 0x{{.*}}
  new_delete<int>(n);

  // CHECK: [Trace] Alloc 0x{{.*}} int64 8 42
  // CHECK: [Trace] Free 0x{{.*}}
  new_delete<long>(n);

  // CHECK: [Trace] Alloc 0x{{.*}} float 4 42
  // CHECK: [Trace] Free 0x{{.*}}
  new_delete<float>(n);

  // CHECK: [Trace] Alloc 0x{{.*}} double 8 42
  // CHECK: [Trace] Free 0x{{.*}}
  new_delete<double>(n);

  // CHECK: [Trace] Alloc 0x{{.*}} pointer 8 42
  // CHECK: [Trace] Free 0x{{.*}}
  new_delete<int*>(n);

  return 0;
}

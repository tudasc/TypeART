// RUN: %run %s --clean_types 2>&1 | %filecheck %s

struct Datastruct {
  int start;
  double middle;
  float end;
};

int main(int argc, char** argv) {
  // CHECK: [Trace] TypeART Runtime Trace
  // CHECK: [Warning]{{.*}}No type file with default name

  // CHECK: [Trace] Alloc [[POINTER:0x[0-9a-fA-F]+]] 256 typeart_unknown_struct 0 1
  struct Datastruct data = {0};
  // CHECK: [Trace] Free [[POINTER]] 256 typeart_unknown_struct 0 1
  return data.start;
}

// RUN: exit 0
// PASS: *

struct X {
  int a;
};

int main(int argc, char** argv) {
  struct X x;
  return x.a;
}
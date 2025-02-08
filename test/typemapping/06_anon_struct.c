// RUN: %c-to-llvm %s | %apply-typeart --typeart-stack=true
// RUN: cat %tu_yaml | %filecheck %s

// REQUIRES: dimeta

struct {
  int dir;
  int length;
  float coeff;
  float forwback;
} q_paths;

struct {
  int dir;
  float length;
  float coeff;
  float forwback;
} q_paths_2;

struct {
  float dir;
  float length;
  float coeff;
  float forwback;
} q_paths_3;

struct {
  struct {
    float dir;
    float length;
    float coeff;
    float forwback;
  } inner;
} q_paths_4;

// CHECK-COUNT-4: name: anonymous_compound_{{[0-9a-z]+}}

// RUN: export TYPEART_INSTRUMENTATION=1

// RUN: %cpp-to-llvm %s | %apply-typeart -typeart-instumentation=true -S | %filecheck --match-full-lines %s

// CHECK: @_typeart__ZTS6Domain = extern_weak constant %struct.typeart_struct_layout_t

// REQUIRES: llvm-18 || llvm-19

class Domain {
 public:
  Domain(int ranks, double other);

  int getRanks() const {
    return m_ranks;
  }

  double getOther() const {
    return m_other;
  }

 private:
  int m_ranks;
  double m_other;
};

int main(int argc, char* argv[]) {
  Domain* dom;

  dom = new Domain(argc, 1.2);

  return dom->getRanks();
}

// RUN: %cpp-to-llvm %s | %apply-typeart -typeart-type-serialization=inline -S | %filecheck %s

// CHECK: @_typeart__ZTS6Domain_fwd = linkonce_odr global %struct._typeart_struct_layout_t { i32 256,

// REQUIRES: !llvm-14

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

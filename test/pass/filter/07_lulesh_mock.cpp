// clang-format off
// RUN: %c-to-llvm -fno-discard-value-names %s | %apply-typeart -typeart-stack -typeart-call-filter -typeart-filter-pointer-alloca=false -S 2>&1 | %filecheck %s --check-prefix=CHECK-exp-default
// RUN: %c-to-llvm -fno-discard-value-names %s | %opt -O3 -S | %apply-typeart -typeart-stack -typeart-call-filter -typeart-filter-pointer-alloca=false -S 2>&1 | %filecheck %s --check-prefix=CHECK-exp-default-opt
// clang-format on

using Real_t = double;
using Int_t  = int;

#define real_f(name) \
  Real_t _##name;    \
  Real_t& name() {   \
    return _##name;  \
  }

#define int_f(name) \
  Int_t _##name;    \
  Int_t& name() {   \
    return _##name; \
  }

struct Domain {
  real_f(stoptime);
  real_f(time);
  real_f(dtfixed);
  real_f(deltatime);
  real_f(dtcourant);
  real_f(dthydro);
  real_f(deltatimemultub);
  real_f(deltatimemultlb);
  real_f(dtmax);
  int_f(cycle);
};

extern void MPI_Allreduce(void*, void*, int);

void TimeIncrement(Domain& domain) {
  Real_t targetdt = domain.stoptime() - domain.time();

  if ((domain.dtfixed() <= Real_t(0.0)) && (domain.cycle() != Int_t(0))) {
    Real_t ratio;
    Real_t olddt = domain.deltatime();

    /* This will require a reduction in parallel */
    Real_t gnewdt = Real_t(1.0e+20);
    Real_t newdt;
    if (domain.dtcourant() < gnewdt) {
      gnewdt = domain.dtcourant() / Real_t(2.0);
    }
    if (domain.dthydro() < gnewdt) {
      gnewdt = domain.dthydro() * Real_t(2.0) / Real_t(3.0);
    }

    MPI_Allreduce(&gnewdt, &newdt, 1);  // Need to keep gnewdt and newdt

    ratio = newdt / olddt;
    if (ratio >= Real_t(1.0)) {
      if (ratio < domain.deltatimemultlb()) {
        newdt = olddt;
      } else if (ratio > domain.deltatimemultub()) {
        newdt = olddt * domain.deltatimemultub();
      }
    }

    if (newdt > domain.dtmax()) {
      newdt = domain.dtmax();
    }
    domain.deltatime() = newdt;
  }

  if ((targetdt > domain.deltatime()) && (targetdt < (Real_t(4.0) * domain.deltatime() / Real_t(3.0)))) {
    targetdt = Real_t(2.0) * domain.deltatime() / Real_t(3.0);
  }

  if (targetdt < domain.deltatime()) {
    domain.deltatime() = targetdt;
  }

  domain.time() += domain.deltatime();

  ++domain.cycle();
}

// Standard experimental filter
// CHECK-exp-default: > Stack Memory
// CHECK-exp-default-NEXT: Alloca                 :  16.00
// CHECK-exp-default-NEXT: Stack call filtered %  :  87.50

// Standard experimental filter with -O3
// CHECK-exp-default-opt: > Stack Memory
// CHECK-exp-default-opt-NEXT: Alloca                 :  2.00
// CHECK-exp-default-opt-NEXT: Stack call filtered %  :  0.00
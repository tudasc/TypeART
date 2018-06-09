// RUN: %scriptpath/applyAndRun.sh %s %pluginpath "-must-alloca" %rtpath | FileCheck %s


template<typename T>
void new_delete() {
    T* t = new T;
    delete t;
}

template<typename T>
void new_delete(int n) {
    T* t = new T[n];
    delete[] t;
}



int main(int argc, char** argv) {
    const int n = 42;

    // CHECK: MUST Support Runtime Trace

    // CHECK: Alloc    0x{{.*}}    char    1   1
    // CHECK: Free 0x{{.*}}
    new_delete<char>();

    // CHECK: Alloc    0x{{.*}}    short    2   1
    // CHECK: Free 0x{{.*}}
    new_delete<short>();

    // CHECK: Alloc    0x{{.*}}    int    4   1
    // CHECK: Free 0x{{.*}}
    new_delete<int>();

    // CHECK: Alloc    0x{{.*}}    long    8   1
    // CHECK: Free 0x{{.*}}
    new_delete<long>();

    // CHECK: Alloc    0x{{.*}}    float    4   1
    // CHECK: Free 0x{{.*}}
    new_delete<float>();

    // CHECK: Alloc    0x{{.*}}    double    8   1
    // CHECK: Free 0x{{.*}}
    new_delete<double>();

    // CHECK: Alloc    0x{{.*}} unknown 8   1
    // CHECK: Free 0x{{.*}}
    new_delete<int*>();


    // CHECK: Alloc    0x{{.*}}    char    1   42
    // CHECK: Free 0x{{.*}}
    new_delete<char>(n);

    // CHECK: Alloc    0x{{.*}}    short    2   42
    // CHECK: Free 0x{{.*}}
    new_delete<short>(n);

    // CHECK: Alloc    0x{{.*}}    int    4   42
    // CHECK: Free 0x{{.*}}
    new_delete<int>(n);

    // CHECK: Alloc    0x{{.*}}    long    8   42
    // CHECK: Free 0x{{.*}}
    new_delete<long>(n);

    // CHECK: Alloc    0x{{.*}}    float    4   42
    // CHECK: Free 0x{{.*}}
    new_delete<float>(n);

    // CHECK: Alloc    0x{{.*}}    double    8   42
    // CHECK: Free 0x{{.*}}
    new_delete<double>(n);

    // CHECK: Alloc    0x{{.*}} unknown 8   42
    // CHECK: Free 0x{{.*}}
    new_delete<int*>(n);


    return 0;
}

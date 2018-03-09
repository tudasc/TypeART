// RUN: clang -S -emit-llvm %s -o - | opt -load %pluginpath/%pluginname %pluginargs -S 2>&1 | FileCheck %s
#include <stdlib.h>
void test() {
    void* p = malloc(42 * sizeof(int));
    int* pi = (int*) p;
    short* ps = (short*) p;
}

// CHECK: [WARNING]
// CHECK: Malloc{{[ ]*}}:{{[ ]*}}1
// Also required (TBD): Alloca{{[ ]*}}:{{[ ]*}}0


define void @test() #0 {
entry:
%p = alloca i8*, align 8
%pi = alloca i32*, align 8
%ps = alloca i16*, align 8
%call = call noalias i8* @malloc(i64 168) #2
store i8* %call, i8** %p, align 8
%0 = load i8*, i8** %p, align 8
%1 = bitcast i8* %0 to i32*
store i32* %1, i32** %pi, align 8
%2 = load i8*, i8** %p, align 8
%3 = bitcast i8* %2 to i16*
store i16* %3, i16** %ps, align 8
ret void
}

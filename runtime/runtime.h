#ifndef RUNTIME_RUNTIME_H_
#define RUNTIME_RUNTIME_H_

extern "C" {
void __must_support_alloc(void* addr, int typeId, long count, long typeSize);
void __must_support_dealloc(void* addr);
}

#endif
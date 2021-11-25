#define _GNU_SOURCE

#include <dlfcn.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <threads.h>
#include <unistd.h>

extern thread_local int8_t TYPEART_CONTEXT;

static void* libc_function(const char* function_name) {
  void* function = dlsym(RTLD_NEXT, function_name);
  if (!function) {
    fprintf(stderr, "Cannot find symbol \'%s\'!\n", function_name);
    abort();
  }
  return function;
}

static thread_local int8_t skip = 0;

void __typeart_preload_print_backtrace() {
  skip += 1;
  int nptrs;
  void* buffer[512];
  char** strings;

  fprintf(stderr, "backtrace()\n");
  nptrs = backtrace(buffer, 512);
  fprintf(stderr, "backtrace() returned %d addresses\n", nptrs);

  /* The call backtrace_symbols_fd(buffer, nptrs, STDOUT_FILENO)
     would produce similar output to the following: */

  backtrace_symbols_fd(buffer, nptrs, STDERR_FILENO);
  skip -= 1;
}

size_t const offset = 32;

void* malloc(size_t size) {
  // fprintf(stderr, "%d MALLOC %d\n", TYPEART_CONTEXT, skip);
  static thread_local void* (*real_malloc)(size_t) = NULL;
  if (!real_malloc) {
    real_malloc = (void* (*)(size_t))libc_function("malloc");
  }
  if (skip >= 1 || TYPEART_CONTEXT == 1) {
    return real_malloc(size);
  }
  skip += 1;
  void* result = real_malloc(size + offset);
  skip -= 1;
  // fprintf(stderr, "==> %d MALLOC %p of size %zu\n", result, size);
  return result == NULL ? NULL : ((char*)result + offset);
}

static thread_local char* buffer[8192] = {0};

void* initial_calloc(size_t num, size_t size) {
  // fprintf(stderr, "INITIAL CALLOC %p\n", buffer);
  return buffer;
}

void* calloc(size_t num, size_t size) {
  // fprintf(stderr, "CALLOC %d\n", skip);
  static thread_local void* (*real_calloc)(size_t, size_t) = NULL;
  if (!real_calloc) {
    real_calloc = initial_calloc;
    real_calloc = (void* (*)(size_t, size_t))libc_function("calloc");
  }
  if (skip >= 1) {
    return real_calloc(num, size);
  }
  size_t bytes = num * size;
  skip += 1;
  void* result = real_calloc(bytes + offset, 1);
  skip -= 1;
  // fprintf(stderr, "==> CALLOC %p of num %zu and size %zu\n", result, num, size);
  return result == NULL ? NULL : ((char*)result + offset);
}

void free(void* ptr) {
  // fprintf(stderr, "%d FREE %d\n", TYPEART_CONTEXT, skip);
  static thread_local void (*real_free)(void*) = NULL;
  if (!real_free) {
    real_free = (void (*)(void*))libc_function("free");
  }
  if (skip >= 1) {
    real_free(ptr);
    return;
  }
  // fprintf(stderr, "==> FREE %p\n", ptr == NULL ? NULL : ((char*)ptr - offset));
  skip += 1;
  real_free(ptr == NULL ? NULL : ((char*)ptr - offset));
  skip -= 1;
}

void* realloc(void* ptr, size_t new_size) {
  // fprintf(stderr, "REALLOC %d\n", skip);
  static thread_local void* (*real_realloc)(void*, size_t) = NULL;
  if (!real_realloc) {
    real_realloc = (void* (*)(void*, size_t))libc_function("realloc");
  }
  if (skip >= 1) {
    return real_realloc(ptr, new_size);
  }
  // fprintf(stderr, "==> REALLOC %p to %zu\n", ptr, new_size);
  skip += 1;
  void* result = real_realloc(ptr == NULL ? NULL : ((char*)ptr - offset), new_size + offset);
  skip -= 1;
  // fprintf(stderr, "==> REALLOC %p of size %zu\n", result, new_size);
  return result == NULL ? NULL : ((char*)result + offset);
}

void* memalign(size_t alignment, size_t size) {
  typedef void* (*ACTUAL)(size_t, size_t);
  static thread_local ACTUAL actual = NULL;
  if (!actual) {
    actual = (ACTUAL)libc_function("memalign");
  }
  void* result = actual(alignment, size);
  // fprintf(stderr, "==> MEMALIGN %p\n", result);
  return result;
}

void* aligned_alloc(size_t alignment, size_t size) {
  typedef void* (*ACTUAL)(size_t, size_t);
  static thread_local ACTUAL actual = NULL;
  if (!actual) {
    actual = (ACTUAL)libc_function("aligned_alloc");
  }
  void* result = actual(alignment, size);
  // fprintf(stderr, "==> ALIGNED_ALLOC %p\n", result);
  return result;
}

void* valloc(size_t size) {
  typedef void* (*ACTUAL)(size_t);
  static thread_local ACTUAL actual = NULL;
  if (!actual) {
    actual = (ACTUAL)libc_function("valloc");
  }
  void* result = actual(size);
  // fprintf(stderr, "==> VALLOC %p\n", result);
  return result;
}

void* pvalloc(size_t size) {
  typedef void* (*ACTUAL)(size_t);
  static thread_local ACTUAL actual = NULL;
  if (!actual) {
    actual = (ACTUAL)libc_function("pvalloc");
  }
  void* result = actual(size);
  // fprintf(stderr, "==> PVALLOC %p\n", result);
  return result;
}

int posix_memalign(void** memptr, size_t alignment, size_t size) {
  typedef int (*ACTUAL)(void**, size_t, size_t);
  static thread_local ACTUAL actual = NULL;
  if (!actual) {
    actual = (ACTUAL)libc_function("posix_memalign");
  }
  int result = actual(memptr, alignment, size);
  // fprintf(stderr, "==> POSIX_MEMALIGN %p\n", *memptr);
  return result;
}

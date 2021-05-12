#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

#define RETURN_ERRORS(expr)            \
  if (auto err = (expr).takeError()) { \
    return {std::move(err)};           \
  }

llvm::Error make_string_error(const char* message) {
  return llvm::make_error<llvm::StringError>(llvm::inconvertibleErrorCode(), message);
}

#define FAIL_IF(condition, message)      \
  if (condition) {                       \
    LOG_FATAL(message);                  \
    return {make_string_error(message)}; \
  }

#define FAIL_IF_FORMATV(condition, format, ...)               \
  if (condition) {                                            \
    std::string message = llvm::formatv(format, __VA_ARGS__); \
    LOG_FATAL(message);                                       \
    return {make_string_error(message.c_str())};              \
  }

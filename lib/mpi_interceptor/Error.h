// TypeART library
//
// Copyright (c) 2017-2021 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#ifndef TYPEART_MPI_INTERCEPTOR_ERROR_H
#define TYPEART_MPI_INTERCEPTOR_ERROR_H

#include "Config.h"
#include "System.h"

#include <memory>
#include <mpi.h>
#include <optional>
#include <result.hpp>
#include <variant>
#include <vector>

namespace typeart {
template <class... Ts>
struct [[nodiscard]] VariantError {
 private:
  std::variant<Ts...> data;

 public:
  template <class... Param>
  VariantError(Param&&... param) : data(std::forward<Param>(param)...) {
  }

  template <class T>
  [[nodiscard]] bool is() const {
    return std::holds_alternative<T>(data);
  }

  template <class T>
  [[nodiscard]] const T& get() const& {
    return std::get<T>(data);
  }

  template <class T>
  [[nodiscard]] T get() && {
    return std::get<T>(std::move(data));
  }

  template <class Visitor>
  auto visit(Visitor&& visitor) const -> decltype(auto) {
    return std::visit(std::forward<Visitor>(visitor), data);
  }
};

struct MPIError {
  std::string function_name;
  std::string message;
};
struct TypeARTError {
  std::string message;
};
struct InvalidArgument {
  std::string message;
};
struct UnsupportedCombiner {
  std::string combiner_name;
};
struct UnsupportedCombinerArgs {
  std::string message;
};
struct [[nodiscard]] InternalError
    : public VariantError<MPIError, TypeARTError, InvalidArgument, UnsupportedCombiner, UnsupportedCombinerArgs> {};

struct TypeError;
struct InsufficientBufferSize {
  size_t actual;
  size_t required;
};
struct BuiltinTypeMismatch {
  int buffer_type_id;
  MPI_Datatype mpi_type;
};
struct BufferNotOfStructType {
  int buffer_type_id;
};
struct MemberCountMismatch {
  int buffer_type_id;
  size_t buffer_count;
  int mpi_count;
};
struct MemberOffsetMismatch {
  int type_id;
  size_t member;
  ptrdiff_t struct_offset;
  MPI_Aint mpi_offset;
};
struct MemberTypeMismatch {
  size_t member;
  std::unique_ptr<TypeError> error;
};
struct MemberElementCountMismatch {
  int type_id;
  size_t member;
  size_t count;
  size_t mpi_count;
};
struct StructSubtypeMismatch {
  int struct_type_id;
  int subtype_id;
  size_t subtype_count;
  std::unique_ptr<TypeError> error;
};
struct StructSubtypeErrors {
  std::unique_ptr<TypeError> primary_error;
  std::vector<StructSubtypeMismatch> subtype_errors;
};
struct [[nodiscard]] TypeError
    : public VariantError<StructSubtypeErrors, InsufficientBufferSize, BuiltinTypeMismatch, BufferNotOfStructType,
                          MemberCountMismatch, MemberOffsetMismatch, MemberTypeMismatch, MemberElementCountMismatch> {};

struct [[nodiscard]] Error : public VariantError<InternalError, TypeError> {
  std::optional<Stacktrace> stacktrace =
      Config::get().with_backtraces ? std::optional{Stacktrace::current()} : std::optional<Stacktrace>{};
};

template <class T>
struct Result : public cpp::result<T, std::shared_ptr<Error>> {
  Result(T value) : cpp::result<T, std::shared_ptr<Error>>(std::move(value)){};

  Result(std::shared_ptr<Error> err) : cpp::result<T, std::shared_ptr<Error>>(cpp::fail(std::move(err))){};

  Result(cpp::result<T, std::shared_ptr<Error>> result) : cpp::result<T, std::shared_ptr<Error>>(std::move(result)){};
};

template <>
struct Result<void> : public cpp::result<void, std::shared_ptr<Error>> {
  Result() : cpp::result<void, std::shared_ptr<Error>>(){};

  Result(std::shared_ptr<Error> err) : cpp::result<void, std::shared_ptr<Error>>(cpp::fail(std::move(err))){};

  Result(cpp::result<void, std::shared_ptr<Error>> result)
      : cpp::result<void, std::shared_ptr<Error>>(std::move(result)){};
};

template <class Type, class... Param>
std::shared_ptr<Error> make_internal_error(Param... param) {
  return std::make_shared<Error>(Error{InternalError{Type{std::forward<Param>(param)...}}});
}

template <class Type, class... Param>
std::shared_ptr<Error> make_type_error(Param... param) {
  return std::make_shared<Error>(Error{TypeError{Type{std::forward<Param>(param)...}}});
}

}  // namespace typeart

#endif  // TYPEART_MPI_INTERCEPTOR_ERROR_H

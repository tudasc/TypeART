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

#include "System.h"

#include <memory>
#include <mpi.h>
#include <result.hpp>
#include <variant>
#include <vector>

namespace typeart {
template <class... Ts>
struct [[nodiscard]] VariantError {
  std::variant<Ts...> data;

  template <class... Param>
  VariantError(Param&&... param) : data(std::forward<Param>(param)...) {
  }

  template <class T>
  [[nodiscard]] bool is() const {
    return std::holds_alternative<T>(data);
  }

  template <class Visitor>
  auto visit(Visitor&& visitor) const -> decltype(auto) {
    return std::visit(std::forward<Visitor>(visitor), data);
  }
};

struct Error;
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
struct InsufficientBufferSize {
  size_t actual;
  size_t required;
};
struct BuiltinTypeMismatch {
  int buffer_type_id;
  MPI_Datatype mpi_type;
};
struct UnsupportedCombinerArgs {
  std::string message;
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
  std::shared_ptr<Error> error;
};
struct MemberElementCountMismatch {
  int type_id;
  size_t member;
  size_t count;
  size_t mpi_count;
};

struct [[nodiscard]] Error
    : public VariantError<MPIError, TypeARTError, InvalidArgument, UnsupportedCombiner, InsufficientBufferSize,
                          BuiltinTypeMismatch, UnsupportedCombinerArgs, BufferNotOfStructType, MemberCountMismatch,
                          MemberOffsetMismatch, MemberTypeMismatch, MemberElementCountMismatch> {
  Stacktrace stacktrace = Stacktrace::current();
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
std::shared_ptr<Error> make_error(Param... param) {
  return std::make_shared<Error>(Error{Type{std::forward<Param>(param)...}});
}

}  // namespace typeart

#endif  // TYPEART_MPI_INTERCEPTOR_ERROR_H

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

#include "Logger.h"

#include <spdlog/pattern_formatter.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>

class LogLevelFlag : public spdlog::custom_flag_formatter {
 public:
  void format(const spdlog::details::log_msg& msg, const std::tm&, spdlog::memory_buf_t& dest) override {
    std::string level;
    switch (msg.level) {
      case spdlog::level::trace:
        level = "Trace";
        break;
      case spdlog::level::debug:
        level = "Debug";
        break;
      case spdlog::level::info:
        level = "Info";
        break;
      case spdlog::level::warn:
        level = "Warning";
        break;
      case spdlog::level::err:
        level = "Error";
        break;
      case spdlog::level::critical:
        level = "Critical";
        break;
      default:
        level = "Unknown";
        break;
    }
    dest.append(level.data(), level.data() + level.size());
  }

  std::unique_ptr<custom_flag_formatter> clone() const override {
    return spdlog::details::make_unique<LogLevelFlag>();
  }
};

class MPIRankFlag : public spdlog::custom_flag_formatter {
  std::string rank = "-";

 public:
  void format(const spdlog::details::log_msg&, const std::tm&, spdlog::memory_buf_t& dest) override {
    int initialized;
    MPI_Initialized(&initialized);
    if (initialized) {
      int mpi_rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
      rank = std::to_string(mpi_rank);
    }
    dest.append(rank.data(), rank.data() + rank.size());
  }

  std::unique_ptr<custom_flag_formatter> clone() const override {
    return spdlog::details::make_unique<MPIRankFlag>();
  }
};

auto create_logger() {
  auto result    = spdlog::stderr_logger_mt("typeart_mpi_interceptor_logger");
  auto formatter = std::make_unique<spdlog::pattern_formatter>();
  formatter->add_flag<MPIRankFlag>('M').add_flag<LogLevelFlag>('L').set_pattern("R[%M][%L]T[%t] %v");
  result->set_formatter(std::move(formatter));
  return result;
}

namespace typeart::logger {

static auto logger = create_logger();

std::string name_for(MPI_Datatype datatype) {
  int len;
  char mpi_type_name[MPI_MAX_OBJECT_NAME];
  int mpierr = MPI_Type_get_name(datatype, &mpi_type_name[0], &len);
  return mpi_type_name;
}

struct Visitor {
  const char* function_name;
  const void* called_from;

  template <class... Ts>
  void print_internal_error(const std::string& fmt, Ts... fmt_args) {
    logger->error("internal error while typechecking a call to {} from {}: " + fmt + "\n", function_name, called_from,
                  std::forward<Ts>(fmt_args)...);
  }
  void operator()(const MPIError& err) {
    print_internal_error("{} failed: {}", err.function_name, err.message);
  }
  void operator()(const TypeARTError& err) {
    print_internal_error("internal runtime error ({})", err.message);
  }
  void operator()(const InvalidArgument& err) {
    print_internal_error("{}", err.message);
  }
  void operator()(const UnsupportedCombiner& err) {
    logger->error("the MPI type combiner {} is currently not supported", err.combiner_name);
  }
  void operator()(const InsufficientBufferSize& err) {
    logger->error("buffer too small ({} elements, {} required)", err.actual, err.required);
  }
  void operator()(const BuiltinTypeMismatch& err) {
    auto type_name     = typeart_get_type_name(err.buffer_type_id);
    auto mpi_type_name = name_for(err.mpi_type);
    logger->error("expected a type matching MPI type \"{}\", but found type \"{}\"", mpi_type_name, type_name);
  }
  void operator()(const UnsupportedCombinerArgs& err) {
    logger->error("{}", err.message);
  }
  void operator()(const BufferNotOfStructType& err) {
    auto type_name = typeart_get_type_name(err.buffer_type_id);
    logger->error("expected a struct type, but found type \"{}\"", type_name);
  }
  void operator()(const MemberCountMismatch& err) {
    auto type_name = typeart_get_type_name(err.buffer_type_id);
    logger->error("expected {} members, but the type \"{}\" has {} members", err.mpi_count, type_name,
                  err.buffer_count);
  }
  void operator()(const MemberOffsetMismatch& err) {
    auto type_name = typeart_get_type_name(err.type_id);
    logger->error("expected a byte offset of {} for member {}, but the type \"{}\" has an offset of {}", err.mpi_offset,
                  err.member, type_name, err.struct_offset);
  }
  void operator()(const MemberTypeMismatch& err) {
    (*err.error).visit(*this);
    logger->error("the typechek for member {} failed", err.member);
  }
  void operator()(const MemberElementCountMismatch& err) {
    auto type_name = typeart_get_type_name(err.type_id);
    logger->error("expected element count of {} for member {}, but the type \"{}\" has a count of {}", err.mpi_count,
                  err.member, type_name, err.count);
  }
};

#ifdef NDEBUG
constexpr auto print_info = false;
#else
constexpr auto print_info = true;
#endif

void null_buffer() {
  logger->warn("buffer is NULL");
}

void result(const char* name, const void* called_from, bool is_send, const Buffer& buffer, const MPIType& type,
            const Result<void>& result) {
  auto type_name     = typeart_get_type_name(buffer.type_id);
  auto mpi_type_name = name_for(type.mpi_type);
  if constexpr (print_info) {
    logger->info("at {}: {}: checking {}-buffer {} of type \"{}\" against MPI type \"{}\"\n", called_from, name,
                 is_send ? "send" : "recv", buffer.ptr, type_name, mpi_type_name);
  }
  if (result.has_error()) {
    (*result.error()).visit(Visitor{name, called_from});
  }
}

void error(const char* name, const void* called_from, const InternalError& error) {
  error.visit(Visitor{name, called_from});
}

void call_counter(const CallCounter& call_counter, long ru_maxrss) {
  logger->info("CCounter {{ Send: {} Recv: {} Send_Recv: {} Unsupported: {} MAX RSS[KBytes]: {} }}", call_counter.send,
               call_counter.recv, call_counter.send_recv, call_counter.unsupported, ru_maxrss);
}

void mpi_counter(const MPICounter& mpi_counter) {
  logger->info("MCounter {{ Error: {} Null_Buf: {} Null_Count: {} Type_Error: {} }}", mpi_counter.error,
               mpi_counter.null_buff, mpi_counter.null_count, mpi_counter.type_error);
}

void unsupported(const char* name) {
  logger->error("The MPI function {} is currently not checked by TypeArt", name);
}

}  // namespace typeart::logger

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

#ifdef NDEBUG
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_ERROR
#else
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#endif

#include "Logger.h"

#include <fmt/ostream.h>
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
}

namespace typeart {

Logger::Logger() {
  logger         = spdlog::stderr_logger_mt("typeart_mpi_interceptor_logger");
  auto formatter = std::make_unique<spdlog::pattern_formatter>();
  formatter->add_flag<MPIRankFlag>('M').add_flag<LogLevelFlag>('L').set_pattern("R[%M][%L]T[%t] %v");
  logger->set_formatter(std::move(formatter));
}

Logger::~Logger(){};

std::string name_for(MPI_Datatype datatype) {
  int len;
  char mpi_type_name[MPI_MAX_OBJECT_NAME];
  int mpierr = MPI_Type_get_name(datatype, &mpi_type_name[0], &len);
  return mpi_type_name;
}

struct InternalErrorVisitor {
  std::string operator()(const MPIError& err) {
    return fmt::format("{} failed: {}", err.function_name, err.message);
  }
  std::string operator()(const TypeARTError& err) {
    return fmt::format("{}", err.message);
  }
  std::string operator()(const InvalidArgument& err) {
    return fmt::format("{}", err.message);
  }
  std::string operator()(const UnsupportedCombiner& err) {
    return fmt::format("the MPI type combiner {} is currently not supported", err.combiner_name);
  }
};

struct TypeErrorVisitor {
  std::string operator()(const InsufficientBufferSize& err) {
    return fmt::format("buffer too small ({} elements, {} required)", err.actual, err.required);
  }
  std::string operator()(const BuiltinTypeMismatch& err) {
    auto type_name     = typeart_get_type_name(err.buffer_type_id);
    auto mpi_type_name = name_for(err.mpi_type);
    return fmt::format("expected a type matching MPI type \"{}\", but found type \"{}\"", mpi_type_name, type_name);
  }
  std::string operator()(const UnsupportedCombinerArgs& err) {
    return fmt::format("{}", err.message);
  }
  std::string operator()(const BufferNotOfStructType& err) {
    auto type_name = typeart_get_type_name(err.buffer_type_id);
    return fmt::format("expected a struct type, but found type \"{}\"", type_name);
  }
  std::string operator()(const MemberCountMismatch& err) {
    auto type_name = typeart_get_type_name(err.buffer_type_id);
    return fmt::format("expected {} members, but the type \"{}\" has {} members", err.mpi_count, type_name,
                       err.buffer_count);
  }
  std::string operator()(const MemberOffsetMismatch& err) {
    auto type_name = typeart_get_type_name(err.type_id);
    return fmt::format("expected a byte offset of {} for member {}, but the type \"{}\" has an offset of {}",
                       err.mpi_offset, err.member, type_name, err.struct_offset);
  }
  std::string operator()(const MemberTypeMismatch& err) {
    return fmt::format("the typechek for member {} failed: {}", err.member, (*err.error).visit(*this));
  }
  std::string operator()(const MemberElementCountMismatch& err) {
    auto type_name = typeart_get_type_name(err.type_id);
    return fmt::format("expected element count of {} for member {}, but the type \"{}\" has a count of {}",
                       err.mpi_count, err.member, type_name, err.count);
  }
};

struct ErrorVisitor {
  std::string operator()(const InternalError& err) {
    return err.visit(InternalErrorVisitor{});
  }
  std::string operator()(const TypeError& err) {
    return err.visit(TypeErrorVisitor{});
  }
};

std::string format_error(const Error& error) {
  return error.visit(ErrorVisitor{});
}

void Logger::log(const char* name, const void* called_from, bool is_send, const Buffer& buffer, const MPIType& type,
                 const Result<void>& result) {
  if (result.has_value()) {
    SPDLOG_LOGGER_INFO(logger, "at {}: {}: checking {}-buffer {} of type \"{}\" against MPI type \"{}\"", called_from,
                       name, is_send ? "send" : "recv", buffer.ptr, typeart_get_type_name(buffer.type_id),
                       name_for(type.mpi_type));
  } else {
    auto error                 = result.error();
    auto internal_error_prefix = error->is<TypeError>() ? "" : "internal error ";
    logger->error("at {}: {}: {}while checking {}-buffer {} of type \"{}\" against MPI type \"{}\": {}", called_from,
                  name, internal_error_prefix, is_send ? "send" : "recv", buffer.ptr,
                  typeart_get_type_name(buffer.type_id), name_for(type.mpi_type), format_error(*error));
  }
}

void Logger::log(const char* name, const void* called_from, bool is_send, const void* ptr, const Error& error) {
  auto error_prefix = error.is<TypeError>() ? "error " : "internal error ";
  logger->error("at {}: {}while checking the {}-buffer {} in a call to {}: {}", called_from, error_prefix, called_from,
                is_send ? "send" : "recv", ptr, name, format_error(error));
}

void Logger::log(const CallCounter& call_counter, long ru_maxrss) {
  logger->info("CCounter {{ Send: {} Recv: {} Send_Recv: {} Unsupported: {} MAX RSS[KBytes]: {} }}", call_counter.send,
               call_counter.recv, call_counter.send_recv, call_counter.unsupported, ru_maxrss);
}

void Logger::log(const MPICounter& mpi_counter) {
  logger->info("MCounter {{ Error: {} Null_Buf: {} Null_Count: {} Type_Error: {} }}", mpi_counter.error,
               mpi_counter.null_buff, mpi_counter.null_count, mpi_counter.type_error);
}

void Logger::log_null_buffer() {
  logger->warn("buffer is NULL");
}

void Logger::log_unsupported(const char* name) {
  logger->error("The MPI function {} is currently not checked by TypeArt", name);
}

}  // namespace typeart

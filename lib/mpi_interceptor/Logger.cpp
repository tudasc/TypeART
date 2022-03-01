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

#if TYPEART_LOG_LEVEL <= 0
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_WARN
#elif TYPEART_LOG_LEVEL <= 2
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_INFO
#else
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#endif

#include "Logger.h"

#include <fmt/ostream.h>
#include <spdlog/pattern_formatter.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>

namespace typeart::logging {

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

  [[nodiscard]] std::unique_ptr<custom_flag_formatter> clone() const override {
    return spdlog::details::make_unique<LogLevelFlag>();
  }
};

class MPIRankFlag : public spdlog::custom_flag_formatter {
  std::string rank = "-";

 public:
  void format(const spdlog::details::log_msg&, const std::tm&, spdlog::memory_buf_t& dest) override {
    int initialized;
    MPI_Initialized(&initialized);

    if (initialized != 0) {
      int mpi_rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
      rank = std::to_string(mpi_rank);
    }

    dest.append(rank.data(), rank.data() + rank.size());
  }

  [[nodiscard]] std::unique_ptr<custom_flag_formatter> clone() const override {
    return spdlog::details::make_unique<MPIRankFlag>();
  }
};
}  // namespace typeart::logging

namespace typeart {

Logger::Logger() {
  using namespace logging;
  logger         = spdlog::stderr_logger_mt("typeart_mpi_interceptor_logger");
  auto formatter = std::make_unique<spdlog::pattern_formatter>();
  formatter->add_flag<MPIRankFlag>('M').add_flag<LogLevelFlag>('L').set_pattern("R[%M][%L]T[%t] %v");
  logger->set_formatter(std::move(formatter));
}

Logger::~Logger() = default;

std::string mpi_name_for(MPI_Datatype datatype) {
  int str_length;
  std::string mpi_type_name;

  mpi_type_name.resize(MPI_MAX_OBJECT_NAME);
  MPI_Type_get_name(datatype, mpi_type_name.data(), &str_length);
  mpi_type_name.resize(str_length);

  return mpi_type_name;
}

std::string mpi_name_or_type_name(const MPIType& mpi_type) {
  std::string name = mpi_name_for(mpi_type.mpi_type);
  if (name.empty()) {
    name = combiner_name_for(mpi_type.combiner.id);
  }
  return name;
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
  std::string operator()(const UnsupportedCombinerArgs& err) {
    return fmt::format("{}", err.message);
  }
};

struct TypeErrorVisitor {
  std::string operator()(const InsufficientBufferSize& err) {
    return fmt::format("buffer too small ({} elements, {} required)", err.actual, err.required);
  }

  std::string operator()(const BuiltinTypeMismatch& err) {
    const auto* type_name = typeart_get_type_name(err.buffer_type_id);
    auto mpi_type_name    = mpi_name_for(err.mpi_type);
    return fmt::format(R"(expected a type matching MPI type "{}", but found type "{}")", mpi_type_name, type_name);
  }

  std::string operator()(const BufferNotOfStructType& err) {
    const auto* type_name = typeart_get_type_name(err.buffer_type_id);
    return fmt::format("expected a struct type, but found type \"{}\"", type_name);
  }

  std::string operator()(const MemberCountMismatch& err) {
    const auto* type_name = typeart_get_type_name(err.buffer_type_id);
    return fmt::format("expected {} members, but the type \"{}\" has {} members", err.mpi_count, type_name,
                       err.buffer_count);
  }

  std::string operator()(const MemberOffsetMismatch& err) {
    const auto* type_name = typeart_get_type_name(err.type_id);
    return fmt::format("expected a byte offset of {} for member {}, but the type \"{}\" has an offset of {}",
                       err.mpi_offset, err.member, type_name, err.struct_offset);
  }

  std::string operator()(const MemberTypeMismatch& err) {
    return fmt::format("the type check for member {} failed ({})", err.member, (*err.error).visit(*this));
  }

  std::string operator()(const MemberElementCountMismatch& err) {
    const auto* type_name = typeart_get_type_name(err.type_id);
    return fmt::format("expected element count of {} for member {}, but the type \"{}\" has a count of {}",
                       err.mpi_count, err.member, type_name, err.count);
  }

  std::string operator()(const StructSubtypeErrors& err) {
    std::vector<std::string> subtype_errors;
    std::transform(err.subtype_errors.begin(), err.subtype_errors.end(), std::back_inserter(subtype_errors),
                   [this](auto&& suberr) {
                     auto struct_type_name = typeart_get_type_name(suberr.struct_type_id);
                     auto type_name        = typeart_get_type_name(suberr.subtype_id);
                     return fmt::format("Tried the first member [{} x {}] of struct type \"{}\" with error: {}",
                                        suberr.subtype_count, type_name, struct_type_name, suberr.error->visit(*this));
                   });
    return fmt::format("{}. {} ]", err.primary_error->visit(*this), fmt::join(subtype_errors, ". "));
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

bool use_source_location_for(spdlog::level::level_enum level) {
  const auto source_location_config = Config::get().getSourceLocation();
  return source_location_config == Config::SourceLocation::None ||
         (source_location_config == Config::SourceLocation::Error && level < spdlog::level::err);
}

std::string format_source_location(spdlog::level::level_enum level, const void* addr) {
  auto source_location =
      use_source_location_for(level) ? std::optional<SourceLocation>{} : SourceLocation::create(addr);

  if (source_location.has_value()) {
    return fmt::format("{}[{}] at {}:{}: ", source_location->function, addr, source_location->file,
                       source_location->line);
  }
  return fmt::format("at {}: ", addr);
}

void Logger::log(const void* called_from, std::string_view prefix, const Error& error) {
  auto source_location = error.stacktrace.has_value() ? "" : format_source_location(spdlog::level::err, called_from);
  logger->error("{}{}{}", source_location, prefix, error.visit(ErrorVisitor{}));

  if (error.stacktrace.has_value()) {
    for (const auto& entry : error.stacktrace.value()) {
      logger->error("\tin {}", entry);
    }
  }
}

void Logger::log(const char* name, const void* called_from, bool is_send, const Buffer& buffer, const MPIType& type,
                 int count, const Result<void>& result) {
  if (result.has_value()) {
    SPDLOG_LOGGER_INFO(logger,
                       "{}{}: successfully checked {}-buffer {} of type [{} x {}] against {} {} of MPI type \"{}\"",
                       format_source_location(spdlog::level::info, called_from), name, is_send ? "send" : "recv",
                       buffer.ptr, buffer.count, typeart_get_type_name(buffer.type_id), count,
                       count == 1 ? "element" : "elements", mpi_name_or_type_name(type));
  } else {
    auto error                        = result.error();
    const auto* internal_error_prefix = error->is<TypeError>() ? "type error " : "internal error ";
    log(called_from,
        fmt::format("{}: {}while checking {}-buffer {} of type [{} x {}] against {} {} of MPI type \"{}\": ", name,
                    internal_error_prefix, is_send ? "send" : "recv", buffer.ptr, buffer.count,
                    typeart_get_type_name(buffer.type_id), count, count == 1 ? "element" : "elements",
                    mpi_name_or_type_name(type)),
        *error);
  }
}

void Logger::log(const char* name, const void* called_from, bool is_send, const void* ptr, const Error& error) {
  const auto* error_prefix = error.is<TypeError>() ? "error " : "internal error ";
  log(called_from,
      fmt::format("{}while checking the {}-buffer {} in a call to {}: ", error_prefix, called_from,
                  is_send ? "send" : "recv", ptr, name),
      error);
}

void Logger::log(const CallCounter& call_counter, long ru_maxrss) {
  SPDLOG_LOGGER_INFO(logger, "CCounter {{ Send: {} Recv: {} Send_Recv: {} Unsupported: {} MAX RSS[KBytes]: {} }}",
                     call_counter.send, call_counter.recv, call_counter.send_recv, call_counter.unsupported, ru_maxrss);
}

void Logger::log(const MPICounter& mpi_counter) {
  SPDLOG_LOGGER_INFO(logger, "MCounter {{ Error: {} Null_Buf: {} Null_Count: {} Type_Error: {} }}", mpi_counter.error,
                     mpi_counter.null_buff, mpi_counter.null_count, mpi_counter.type_error);
}

void Logger::log_zero_count(const char* function_name, const void* called_from, bool is_send, const void* ptr) {
  SPDLOG_LOGGER_WARN(logger, "{}{}: attempted to {} 0 elements of buffer {}",
                     format_source_location(spdlog::level::warn, called_from), function_name,
                     is_send ? "send" : "receive", ptr);
}

void Logger::log_null_buffer(const char* function_name, const void* called_from, bool is_send) {
  SPDLOG_LOGGER_WARN(logger, "{}{}: {}-buffer is NULL", format_source_location(spdlog::level::warn, called_from),
                     function_name, is_send ? "send" : "recv");
}

void Logger::log_unsupported(const char* name) {
  logger->error("The MPI function {} is currently not checked by TypeArt", name);
}

}  // namespace typeart

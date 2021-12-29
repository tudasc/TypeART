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

#include "Stats.h"
#include "TypeCheck.h"

#include <atomic>
#include <memory>

namespace spdlog {
class logger;
}

namespace typeart {

class Logger {
  std::shared_ptr<spdlog::logger> logger;

 public:
  Logger();
  ~Logger();

  void log(const char* name, const void* called_from, bool is_send, const Buffer& buffer, const MPIType& type,
           int count, const Result<void>&);
  void log(const char* function_name, const void* called_from, bool is_send, const void* ptr, const Error&);
  void log(const CallCounter& call_counter, long ru_maxrss);
  void log(const MPICounter& mpi_counter);
  void log_zero_count(const char* function_name, const void* called_from, bool is_send, const void* ptr);
  void log_null_buffer(const char* function_name, const void* called_from, bool is_send);
  void log_unsupported(const char* name);

 private:
  void log(const void* called_from, std::string info, const Error&);
};

}  // namespace typeart

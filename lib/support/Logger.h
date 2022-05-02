// TypeART library
//
// Copyright (c) 2017-2022 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#ifndef LIB_LOGGER_H_
#define LIB_LOGGER_H_

#include "llvm/Support/raw_ostream.h"

#ifndef TYPEART_LOG_LEVEL
/*
 * Usually set at compile time: -DTYPEART_LOG_LEVEL=<N>, N in [0, 3] for output
 * 3 being most verbose
 */
#define TYPEART_LOG_LEVEL 3
#endif

#ifndef LOG_BASENAME_FILE
#define LOG_BASENAME_FILE __FILE__
#endif

#ifndef TYPEART_MPI_LOGGER
#define TYPEART_MPI_LOGGER 0
#endif

namespace typeart::detail {
#if TYPEART_MPI_LOGGER
void typeart_log(const std::string& rank);
#else
inline void typeart_log(const std::string& msg) {
  llvm::errs() << msg;
}
#endif
}  // namespace typeart::detail

// clang-format off
#define OO_LOG_LEVEL_MSG(LEVEL_NUM, LEVEL, MSG)                                                                   \
  if ((LEVEL_NUM) <= TYPEART_LOG_LEVEL) {                                                                                 \
    std::string logging_message;                                                                                  \
    llvm::raw_string_ostream rso(logging_message);                                                                \
    rso << (LEVEL) << LOG_BASENAME_FILE << ":" << __func__ << ":" << __LINE__ << ":" << MSG << "\n"; /* NOLINT */ \
    typeart::detail::typeart_log(rso.str());                                                                      \
  }

#define OO_LOG_LEVEL_MSG_BARE(LEVEL_NUM, LEVEL, MSG)   \
  if ((LEVEL_NUM) <= TYPEART_LOG_LEVEL) {                      \
    std::string logging_message;                       \
    llvm::raw_string_ostream rso(logging_message);     \
    rso << (LEVEL) << " " << MSG << "\n"; /* NOLINT */ \
    typeart::detail::typeart_log(rso.str());           \
  }
// clang-format on

#define LOG_TRACE(MSG)   OO_LOG_LEVEL_MSG_BARE(3, "[Trace]", MSG)
#define LOG_DEBUG(MSG)   OO_LOG_LEVEL_MSG(3, "[Debug]", MSG)
#define LOG_INFO(MSG)    OO_LOG_LEVEL_MSG(2, "[Info]", MSG)
#define LOG_WARNING(MSG) OO_LOG_LEVEL_MSG(1, "[Warning]", MSG)
#define LOG_ERROR(MSG)   OO_LOG_LEVEL_MSG(1, "[Error]", MSG)
#define LOG_FATAL(MSG)   OO_LOG_LEVEL_MSG(0, "[Fatal]", MSG)
#define LOG_MSG(MSG)     llvm::errs() << MSG << "\n"; /* NOLINT */

#endif /* LIB_LOGGER_H_ */

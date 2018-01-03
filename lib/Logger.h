/*
 * Logger.h
 *
 *  Created on: Jan 3, 2018
 *      Author: ahueck
 */

#ifndef LIB_LOGGER_H_
#define LIB_LOGGER_H_

#include <llvm/Support/raw_ostream.h>

#ifndef LOG_LEVEL
/*
 * Usually set at compile time: -DLOG_LEVEL=<N>, N in [0, 3] for output
 * 3 being most verbose
 */
#define LOG_LEVEL 3
#endif

// clang-format off
#define OO_LOG_LEVEL_MSG(LEVEL_NUM, LEVEL, MSG) \
  if ((LEVEL_NUM) <= LOG_LEVEL) { \
    llvm::errs() << (LEVEL) << " " << __FILE__ << ":" << __func__ << ":" << __LINE__ << ": " << MSG << "\n"; \
  }

#define LOG_DEBUG(MSG) OO_LOG_LEVEL_MSG(3, "[Debug]", MSG)
#define LOG_INFO(MSG) OO_LOG_LEVEL_MSG(2, "[Info]", MSG)
#define LOG_ERROR(MSG) OO_LOG_LEVEL_MSG(1, "[Error]", MSG)
#define LOG_FATAL(MSG) OO_LOG_LEVEL_MSG(0, "[Fatal]", MSG)
#define LOG_MSG(MSG) llvm::outs() << MSG << "\n";
// clang-format on

#endif /* LIB_LOGGER_H_ */

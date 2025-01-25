#ifndef D8AEC7EC_5687_4E81_B6E5_97074AEF6D94
#define D8AEC7EC_5687_4E81_B6E5_97074AEF6D94

#include "TypeARTOptions.h"
#include "Configuration.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace typeart::config::pass {

using PassConfig = std::pair<llvm::Expected<TypeARTConfigOptions>, OptOccurrenceMap>;

llvm::Expected<TypeARTConfigOptions> parse_typeart_config(llvm::StringRef parameters);
PassConfig parse_typeart_config_with_occurrence(llvm::StringRef parameters);

}  // namespace typeart::config::pass

#endif /* D8AEC7EC_5687_4E81_B6E5_97074AEF6D94 */

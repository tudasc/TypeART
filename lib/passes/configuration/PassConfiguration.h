#ifndef D8AEC7EC_5687_4E81_B6E5_97074AEF6D94
#define D8AEC7EC_5687_4E81_B6E5_97074AEF6D94

#include "TypeARTOptions.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace typeart::config::pass {

llvm::Expected<TypeARTConfigOptions> parse_typeart_config(llvm::StringRef parameters);

}

#endif /* D8AEC7EC_5687_4E81_B6E5_97074AEF6D94 */

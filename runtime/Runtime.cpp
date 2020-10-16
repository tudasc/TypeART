#include "Runtime.h"

#include "AccessCounter.h"
#include "Logger.h"
#include "RuntimeInterface.h"
#include "TypeIO.h"

#include "llvm/ADT/Optional.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <sstream>
#include <string>

#ifdef USE_BTREE
using namespace btree;
#endif

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

namespace typeart {
/**
 * Ensures that memory tracking functions do not come from within the runtime.
 * TODO: Problematic with respect to future thread safety considerations (also, globals are ugly)
 */
static bool typeart_rt_scope{false};
}  // namespace typeart

#define RUNTIME_GUARD_BEGIN        \
  if (typeart::typeart_rt_scope) { \
    return;                        \
  }                                \
  typeart::typeart_rt_scope = true
#define RUNTIME_GUARD_END typeart::typeart_rt_scope = false

namespace typeart {

// std::string TypeArtRT::defaultTypeFileName{"types.yaml"};

template <typename T>
inline const void* addByteOffset(const void* addr, T offset) {
  return static_cast<const void*>(static_cast<const uint8_t*>(addr) + offset);
}

namespace debug {
inline static std::string toString(const void* memAddr, int typeId, size_t count, size_t typeSize,
                                   const void* calledFrom) {
  std::string buf;
  llvm::raw_string_ostream s(buf);
  const auto name = TypeArtRT::get().getTypeName(typeId);
  s << memAddr << " " << typeId << " " << name << " " << typeSize << " " << count << " (" << calledFrom << ")";
  return s.str();
}

inline static std::string toString(const void* memAddr, int typeId, size_t count, const void* calledFrom) {
  const auto typeSize = TypeArtRT::get().getTypeSize(typeId);
  return toString(memAddr, typeId, count, typeSize, calledFrom);
}

inline static std::string toString(const void* addr, const PointerInfo& info) {
  return toString(addr, info.typeId, info.count, info.debug);
}

void printTraceStart() {
  LOG_TRACE("TypeART Runtime Trace");
  LOG_TRACE("*********************");
  LOG_TRACE("Operation  Address   Type   Size   Count   (CallAddr)   Stack/Heap/Global");
  LOG_TRACE("-------------------------------------------------------------------------");
}
}  // namespace debug

namespace detail {
template <class...>
constexpr std::false_type always_false{};
}

template <typename Enum>
inline Enum operator|(Enum lhs, Enum rhs) {
  if constexpr (std::is_enum_v<Enum> &&
                (std::is_same_v<Enum, TypeArtRT::AllocState> || std::is_same_v<Enum, TypeArtRT::FreeState>)) {
    using enum_t = typename std::underlying_type<Enum>::type;
    return static_cast<Enum>(static_cast<enum_t>(lhs) | static_cast<enum_t>(rhs));
  } else {
    static_assert(detail::always_false<Enum>);
  }
}
template <typename Enum>
inline void operator|=(Enum& lhs, Enum rhs) {
  if constexpr (std::is_enum_v<Enum> &&
                (std::is_same_v<Enum, TypeArtRT::AllocState> || std::is_same_v<Enum, TypeArtRT::FreeState>)) {
    lhs = lhs | rhs;
  } else {
    static_assert(detail::always_false<Enum>);
  }
}

template <typename Enum>
inline Enum operator&(Enum lhs, Enum rhs) {
  if constexpr (std::is_enum_v<Enum> && std::is_same_v<Enum, TypeArtRT::AllocState>) {
    using enum_t = typename std::underlying_type<Enum>::type;
    return static_cast<Enum>(static_cast<enum_t>(lhs) & static_cast<enum_t>(rhs));
  } else {
    static_assert(detail::always_false<Enum>);
  }
}

template <typename Enum>
inline typename std::underlying_type<Enum>::type operator==(Enum lhs, Enum rhs) {
  if constexpr (std::is_enum_v<Enum> && std::is_same_v<Enum, TypeArtRT::AllocState>) {
    using enum_t = typename std::underlying_type<Enum>::type;
    return static_cast<enum_t>(lhs) & static_cast<enum_t>(rhs);
  } else {
    static_assert(detail::always_false<Enum>);
  }
}

using namespace debug;

TypeArtRT::TypeArtRT(Recorder& counter) : counter(counter) {
  // Try to load types from specified file first.
  // Then look at default location.
  const char* typeFile = std::getenv("TA_TYPE_FILE");
  if (typeFile != nullptr) {
    if (!loadTypes(typeFile)) {
      LOG_FATAL("Failed to load recorded types from " << typeFile);
      std::exit(EXIT_FAILURE);  // TODO: Error handling
    }
  } else {
    if (!loadTypes(defaultTypeFileName)) {
      LOG_FATAL("No type file with default name \""
                << defaultTypeFileName
                << "\" in current directory. To specify a different file, edit the TA_TYPE_FILE environment variable.");
      std::exit(EXIT_FAILURE);  // TODO: Error handling
    }
  }

  std::stringstream ss;
  for (const auto& structInfo : typeDB.getStructList()) {
    ss << structInfo.name << ", ";
  }
  LOG_INFO("Recorded types: " << ss.str());

  stackVars.reserve(RuntimeT::StackReserve);

  debug::printTraceStart();
}

TypeArtRT::~TypeArtRT() {
  std::string stats;
  llvm::raw_string_ostream stream(stats);
  softcounter::serialise(counter, stream);
  if (!stream.str().empty()) {
    // llvm::errs/LOG will crash with virtual call error
    std::cerr << stream.str();
  }
}

bool TypeArtRT::loadTypes(const std::string& file) {
  TypeIO cio(typeDB);
  return cio.load(file);
}

llvm::Optional<RuntimeT::MapEntry> TypeArtRT::findBaseAddress(const void* addr) const {
  if (typeMap.empty() || addr < typeMap.begin()->first) {
    return llvm::None;
  }

  auto it = typeMap.lower_bound(addr);
  if (it == typeMap.end()) {
    // No element bigger than base address
    return {*typeMap.rbegin()};
  }

  if (it->first == addr) {
    // Exact match
    return {*it};
  }
  // Base address
  return {*std::prev(it)};
}

size_t TypeArtRT::getMemberIndex(typeart_struct_layout structInfo, size_t offset) const {
  size_t n = structInfo.len;
  if (offset > structInfo.offsets[n - 1]) {
    return n - 1;
  }

  size_t i = 0;
  while (i < n - 1 && offset >= structInfo.offsets[i + 1]) {
    ++i;
  }
  return i;
}

TypeArtRT::TypeArtStatus TypeArtRT::getSubTypeInfo(const void* baseAddr, size_t offset,
                                                   typeart_struct_layout containerInfo, int* subType,
                                                   const void** subTypeBaseAddr, size_t* subTypeOffset,
                                                   size_t* subTypeCount) const {
  if (offset >= containerInfo.extent) {
    return TA_BAD_OFFSET;
  }

  // Get index of the struct member at the address
  size_t memberIndex = getMemberIndex(containerInfo, offset);

  int memberType = containerInfo.member_types[memberIndex];

  size_t baseOffset = containerInfo.offsets[memberIndex];
  assert(offset >= baseOffset && "Invalid offset values");

  size_t internalOffset   = offset - baseOffset;
  size_t typeSize         = typeDB.getTypeSize(memberType);
  size_t offsetInTypeSize = internalOffset / typeSize;
  size_t newOffset        = internalOffset % typeSize;

  // If newOffset != 0, the subtype cannot be atomic, i.e. must be a struct
  if (newOffset != 0) {
    if (typeDB.isReservedType(memberType)) {
      return TA_BAD_ALIGNMENT;
    }
  }

  // Ensure that the array index is in bounds
  if (offsetInTypeSize >= containerInfo.count[memberIndex]) {
    // Address points to padding
    return TA_BAD_ALIGNMENT;
  }

  *subType         = memberType;
  *subTypeBaseAddr = addByteOffset(baseAddr, baseOffset);
  *subTypeOffset   = newOffset;
  *subTypeCount    = containerInfo.count[memberIndex] - offsetInTypeSize;

  return TA_OK;
}

TypeArtRT::TypeArtStatus TypeArtRT::getSubTypeInfo(const void* baseAddr, size_t offset,
                                                   const StructTypeInfo& containerInfo, int* subType,
                                                   const void** subTypeBaseAddr, size_t* subTypeOffset,
                                                   size_t* subTypeCount) const {
  typeart_struct_layout structLayout;
  structLayout.id           = containerInfo.id;
  structLayout.name         = containerInfo.name.c_str();
  structLayout.len          = containerInfo.numMembers;
  structLayout.extent       = containerInfo.extent;
  structLayout.offsets      = &containerInfo.offsets[0];
  structLayout.member_types = &containerInfo.memberTypes[0];
  structLayout.count        = &containerInfo.arraySizes[0];
  return getSubTypeInfo(baseAddr, offset, structLayout, subType, subTypeBaseAddr, subTypeOffset, subTypeCount);
}

TypeArtRT::TypeArtStatus TypeArtRT::getTypeInfoInternal(const void* baseAddr, size_t offset,
                                                        const StructTypeInfo& containerInfo, int* type,
                                                        size_t* count) const {
  assert(offset < containerInfo.extent && "Something went wrong with the base address computation");

  TypeArtStatus status;
  int subType;
  const void* subTypeBaseAddr;
  size_t subTypeOffset;
  size_t subTypeCount;
  const StructTypeInfo* structInfo = &containerInfo;

  bool resolve = true;

  // Resolve type recursively, until the address matches exactly
  while (resolve) {
    status = getSubTypeInfo(baseAddr, offset, *structInfo, &subType, &subTypeBaseAddr, &subTypeOffset, &subTypeCount);

    if (status != TA_OK) {
      return status;
    }

    baseAddr = subTypeBaseAddr;
    offset   = subTypeOffset;

    // Continue as long as there is a byte offset
    resolve = offset != 0;

    // Get layout of the nested struct
    if (resolve) {
      status = getStructInfo(subType, &structInfo);
      if (status != TA_OK) {
        return status;
      }
    }
  }
  *type  = subType;
  *count = subTypeCount;
  return TA_OK;
}

TypeArtRT::TypeArtStatus TypeArtRT::getTypeInfo(const void* addr, int* type, size_t* count) const {
  int containingType;
  size_t containingTypeCount;
  const void* baseAddr;
  size_t internalOffset;

  // First, retrieve the containing type
  TypeArtStatus status = getContainingTypeInfo(addr, &containingType, &containingTypeCount, &baseAddr, &internalOffset);
  if (status != TA_OK) {
    if (status == TA_UNKNOWN_ADDRESS) {
      typeart::Recorder::get().incAddrMissing(addr);
    }
    return status;
  }

  // Check for exact address match
  if (internalOffset == 0) {
    *type  = containingType;
    *count = containingTypeCount;
    return TA_OK;
  }

  if (typeDB.isBuiltinType(containingType)) {
    // Address points to the middle of a builtin type
    return TA_BAD_ALIGNMENT;
  }

  // Resolve struct recursively
  const auto* structInfo = typeDB.getStructInfo(containingType);
  if (structInfo != nullptr) {
    const void* containingTypeAddr = addByteOffset(addr, -internalOffset);
    return getTypeInfoInternal(containingTypeAddr, internalOffset, *structInfo, type, count);
  }
  return TA_INVALID_ID;
}

TypeArtRT::TypeArtStatus TypeArtRT::getContainingTypeInfo(const void* addr, int* type, size_t* count,
                                                          const void** baseAddress, size_t* offset) const {
  // Find the start address of the containing buffer
  auto ptrData = findBaseAddress(addr);

  if (ptrData) {
    const auto& basePtrInfo = ptrData.getValue().second;
    const auto* basePtr     = ptrData.getValue().first;
    size_t typeSize         = getTypeSize(basePtrInfo.typeId);

    // Check for exact match -> no further checks and offsets calculations needed
    if (basePtr == addr) {
      *type        = basePtrInfo.typeId;
      *count       = basePtrInfo.count;
      *baseAddress = addr;
      *offset      = 0;
      return TA_OK;
    }

    // The address points inside a known array
    const void* blockEnd = addByteOffset(basePtr, basePtrInfo.count * typeSize);

    // Ensure that the given address is in bounds and points to the start of an element
    if (addr >= blockEnd) {
      const std::ptrdiff_t offset = static_cast<const uint8_t*>(addr) - static_cast<const uint8_t*>(basePtr);
      const auto oob_index        = (offset / typeSize) - basePtrInfo.count + 1;
      LOG_ERROR("Out of bounds for the lookup: (" << toString(addr, basePtrInfo)
                                                  << ") #Elements too far: " << oob_index);
      return TA_UNKNOWN_ADDRESS;
    }

    assert(addr >= basePtr && "Error in base address computation");
    size_t addrDif = reinterpret_cast<size_t>(addr) - reinterpret_cast<size_t>(basePtr);

    // Offset of the pointer w.r.t. the start of the containing type
    size_t internalOffset = addrDif % typeSize;

    // Array index
    size_t typeOffset = addrDif / typeSize;
    size_t typeCount  = basePtrInfo.count - typeOffset;

    // Retrieve and return type information
    *type        = basePtrInfo.typeId;
    *count       = typeCount;
    *baseAddress = basePtr;  // addByteOffset(basePtr, typeOffset * basePtrInfo.typeSize);
    *offset      = internalOffset;
    return TA_OK;
  }
  return TA_UNKNOWN_ADDRESS;
}

TypeArtRT::TypeArtStatus TypeArtRT::getBuiltinInfo(const void* addr, typeart::BuiltinType* type) const {
  int id;
  size_t count;
  TypeArtStatus result = getTypeInfo(addr, &id, &count);
  if (result == TA_OK) {
    if (typeDB.isReservedType(id)) {
      *type = static_cast<BuiltinType>(id);
      return TA_OK;
    }
    return TA_WRONG_KIND;
  }
  return result;
}

TypeArtRT::TypeArtStatus TypeArtRT::getStructInfo(int id, const StructTypeInfo** structInfo) const {
  // Requested ID must correspond to a struct
  if (!typeDB.isStructType(id)) {
    return TA_WRONG_KIND;
  }

  const auto* result = typeDB.getStructInfo(id);

  if (result != nullptr) {
    *structInfo = result;
    return TA_OK;
  }
  return TA_INVALID_ID;
}

const std::string& TypeArtRT::getTypeName(int id) const {
  return typeDB.getTypeName(id);
}

size_t TypeArtRT::getTypeSize(int id) const {
  return typeDB.getTypeSize(id);
}

void TypeArtRT::getReturnAddress(const void* addr, const void** retAddr) const {
  auto basePtr = findBaseAddress(addr);

  if (basePtr) {
    *retAddr = basePtr.getValue().second.debug;
  } else {
    *retAddr = nullptr;
  }
}

TypeArtRT::AllocState TypeArtRT::doAlloc(const void* addr, int typeId, size_t count, const void* retAddr) {
  AllocState status = AllocState::NO_INIT;
  if (unlikely(!typeDB.isValid(typeId))) {
    status |= AllocState::UNKNOWN_ID;
    LOG_ERROR("Allocation of unknown type " << toString(addr, typeId, count, retAddr));
  }

  // Calling malloc with size 0 may return a nullptr or some address that can not be written to.
  // In the second case, the allocation is tracked anyway so that onFree() does not report an error.
  // On the other hand, an allocation on address 0x0 with size > 0 is an actual error.
  if (unlikely(count == 0)) {
    Recorder::get().incZeroLengthAddr();
    status |= AllocState::ZERO_COUNT;
    LOG_WARNING("Zero-size allocation " << toString(addr, typeId, count, retAddr));
    if (addr == nullptr) {
      Recorder::get().incZeroLengthAndNullAddr();
      LOG_ERROR("Zero-size and nullptr allocation " << toString(addr, typeId, count, retAddr));
      return status | AllocState::NULL_ZERO | AllocState::ADDR_SKIPPED;
    }
  } else if (unlikely(addr == nullptr)) {
    Recorder::get().incNullAddr();
    LOG_ERROR("Zero-size allocation " << toString(addr, typeId, count, retAddr));
    return status | AllocState::NULL_PTR | AllocState::ADDR_SKIPPED;
  }

  auto& def = typeMap[addr];

  if (unlikely(def.typeId != -1)) {
    typeart::Recorder::get().incAddrReuse();
    status |= AllocState::ADDR_REUSE;
    LOG_WARNING("Pointer already in map " << toString(addr, typeId, count, retAddr));
    LOG_WARNING("Overriden data in map " << toString(addr, def));
  }

  def.typeId = typeId;
  def.count  = count;
  def.debug  = retAddr;

  return status | AllocState::OK;
}

void TypeArtRT::onAlloc(const void* addr, int typeId, size_t count, const void* retAddr) {
  const auto status = doAlloc(addr, typeId, count, retAddr);
  if (status != AllocState::ADDR_SKIPPED) {
    typeart::Recorder::get().incHeapAlloc(typeId, count);
  }
  LOG_TRACE("Alloc " << toString(addr, typeId, count, retAddr) << " " << 'H');
}

void TypeArtRT::onAllocStack(const void* addr, int typeId, size_t count, const void* retAddr) {
  const auto status = doAlloc(addr, typeId, count, retAddr);
  if (status != AllocState::ADDR_SKIPPED) {
    stackVars.push_back(addr);
    typeart::Recorder::get().incStackAlloc(typeId, count);
  }
  LOG_TRACE("Alloc " << toString(addr, typeId, count, retAddr) << " " << 'S');
}

void TypeArtRT::onAllocGlobal(const void* addr, int typeId, size_t count, const void* retAddr) {
  const auto status = doAlloc(addr, typeId, count, retAddr);
  if (status != AllocState::ADDR_SKIPPED) {
    typeart::Recorder::get().incGlobalAlloc(typeId, count);
  }
  LOG_TRACE("Alloc " << toString(addr, typeId, count, retAddr) << " " << 'G');
}

template <bool stack>
TypeArtRT::FreeState TypeArtRT::doFree(const void* addr, const void* retAddr) {
  if (unlikely(addr == nullptr)) {
    LOG_ERROR("Free on nullptr "
              << "(" << retAddr << ")");
    return FreeState::ADDR_SKIPPED | FreeState::NULL_PTR;
  }

  const auto it = typeMap.find(addr);

  if (likely(it != typeMap.end())) {
    LOG_TRACE("Free " << toString((*it).first, (*it).second));

    if constexpr (!std::is_same_v<Recorder, softcounter::NoneRecorder>) {
      const auto typeId = it->second.typeId;
      const auto count  = it->second.count;
      if (stack) {
        Recorder::get().incStackFree(typeId, count);
      } else {
        Recorder::get().incHeapFree(typeId, count);
      }
    }

    typeMap.erase(it);
  } else {
    LOG_ERROR("Free on unregistered address " << addr << " (" << retAddr << ")");
    return FreeState::ADDR_SKIPPED | FreeState::UNREG_ADDR;
  }

  return FreeState::OK;
}

void TypeArtRT::onFreeHeap(const void* addr, const void* retAddr) {
  const auto status = doFree<false>(addr, retAddr);
  if (FreeState::OK == status) {
    typeart::Recorder::get().decHeapAlloc();
  }
}

void TypeArtRT::onLeaveScope(size_t alloca_count, const void* retAddr) {
  if (alloca_count > stackVars.size()) {
    LOG_ERROR("Stack is smaller than requested de-allocation count. alloca_count: " << alloca_count
                                                                                    << ". size: " << stackVars.size());
    alloca_count = stackVars.size();
  }

  const auto cend      = stackVars.cend();
  const auto start_pos = (cend - alloca_count);
  LOG_TRACE("Freeing stack (" << alloca_count << ")  " << std::distance(start_pos, stackVars.cend()))
  std::for_each(start_pos, cend, [this, &retAddr](const void* addr) { doFree<true>(addr, retAddr); });
  stackVars.erase(start_pos, cend);
  typeart::Recorder::get().decStackAlloc(alloca_count);
  LOG_TRACE("Stack after free: " << stackVars.size());
}

}  // namespace typeart

void __typeart_alloc(const void* addr, int typeId, size_t count) {
  RUNTIME_GUARD_BEGIN;
  const void* retAddr = __builtin_return_address(0);
  typeart::TypeArtRT::get().onAlloc(addr, typeId, count, retAddr);
  RUNTIME_GUARD_END;
}

void __typeart_alloc_stack(const void* addr, int typeId, size_t count) {
  RUNTIME_GUARD_BEGIN;
  const void* retAddr = __builtin_return_address(0);
  typeart::TypeArtRT::get().onAllocStack(addr, typeId, count, retAddr);
  RUNTIME_GUARD_END;
}

void __typeart_alloc_global(const void* addr, int typeId, size_t count) {
  RUNTIME_GUARD_BEGIN;
  const void* retAddr = __builtin_return_address(0);
  typeart::TypeArtRT::get().onAllocGlobal(addr, typeId, count, retAddr);
  RUNTIME_GUARD_END;
}

void __typeart_free(const void* addr) {
  RUNTIME_GUARD_BEGIN;
  const void* retAddr = __builtin_return_address(0);
  typeart::TypeArtRT::get().onFreeHeap(addr, retAddr);
  RUNTIME_GUARD_END;
}

void __typeart_leave_scope(size_t alloca_count) {
  RUNTIME_GUARD_BEGIN;
  const void* retAddr = __builtin_return_address(0);
  typeart::TypeArtRT::get().onLeaveScope(alloca_count, retAddr);
  RUNTIME_GUARD_END;
}

typeart_status typeart_get_builtin_type(const void* addr, typeart::BuiltinType* type) {
  return typeart::TypeArtRT::get().getBuiltinInfo(addr, type);
}

typeart_status typeart_get_type(const void* addr, int* type, size_t* count) {
  typeart::Recorder::get().incUsedInRequest(addr);
  return typeart::TypeArtRT::get().getTypeInfo(addr, type, count);
}

typeart_status typeart_get_containing_type(const void* addr, int* type, size_t* count, const void** base_address,
                                           size_t* offset) {
  return typeart::TypeArtRT::get().getContainingTypeInfo(addr, type, count, base_address, offset);
}

typeart_status typeart_get_subtype(const void* base_addr, size_t offset, typeart_struct_layout container_layout,
                                   int* subtype, const void** subtype_base_addr, size_t* subtype_offset,
                                   size_t* subtype_count) {
  return typeart::TypeArtRT::get().getSubTypeInfo(base_addr, offset, container_layout, subtype, subtype_base_addr,
                                                  subtype_offset, subtype_count);
}

typeart_status typeart_resolve_type(int id, typeart_struct_layout* struct_layout) {
  const typeart::StructTypeInfo* structInfo;
  typeart_status status = typeart::TypeArtRT::get().getStructInfo(id, &structInfo);
  if (status == TA_OK) {
    struct_layout->id           = structInfo->id;
    struct_layout->name         = structInfo->name.c_str();
    struct_layout->len          = structInfo->numMembers;
    struct_layout->extent       = structInfo->extent;
    struct_layout->offsets      = &structInfo->offsets[0];
    struct_layout->member_types = &structInfo->memberTypes[0];
    struct_layout->count        = &structInfo->arraySizes[0];
  }
  return status;
}

const char* typeart_get_type_name(int id) {
  return typeart::TypeArtRT::get().getTypeName(id).c_str();
}

void typeart_get_return_address(const void* addr, const void** retAddr) {
  return typeart::TypeArtRT::get().getReturnAddress(addr, retAddr);
}

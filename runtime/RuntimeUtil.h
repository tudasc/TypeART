#ifndef TYPEART_RUNTIMEUTIL_H
#define TYPEART_RUNTIMEUTIL_H

namespace typeart {

template <typename T>
inline const void* addByteOffset(const void* addr, T offset) {
  return static_cast<const void*>(static_cast<const uint8_t*>(addr) + offset);
}

}  // namespace typeart

#endif  // TYPEART_RUNTIMEUTIL_H

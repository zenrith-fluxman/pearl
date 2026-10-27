#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Reads word i of the 32-byte BLAKE3 key from global memory as a little-endian
// u32. Read bytewise so the key tensor needs no particular alignment.
__device__ __forceinline__ uint32_t load_key_word(const uint8_t* key, int i) {
  return static_cast<uint32_t>(__ldg(key + 4 * i)) |
         (static_cast<uint32_t>(__ldg(key + 4 * i + 1)) << 8) |
         (static_cast<uint32_t>(__ldg(key + 4 * i + 2)) << 16) |
         (static_cast<uint32_t>(__ldg(key + 4 * i + 3)) << 24);
}

// Loads the 8 words of the BLAKE3 key into registers.
__device__ __forceinline__ void load_key(const uint8_t* key,
                                         uint32_t (&rKey)[8]) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    rKey[i] = load_key_word(key, i);
  }
}

// Writes word i of a 32-byte digest to global memory as little-endian bytes,
// so the output tensor needs no particular alignment.
__device__ __forceinline__ void store_digest_word(uint8_t* out, int i,
                                                  uint32_t word) {
  out[4 * i] = static_cast<uint8_t>(word);
  out[4 * i + 1] = static_cast<uint8_t>(word >> 8);
  out[4 * i + 2] = static_cast<uint8_t>(word >> 16);
  out[4 * i + 3] = static_cast<uint8_t>(word >> 24);
}

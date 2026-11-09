// Shared helpers of the native gfx1030 kernels (same definitions as native-c512/c512.hip, kept here so the ViT /
// decoder modules build without that file). Layouts: native-mid/RESULTS.md.
//  activation tile: 4x4 pixels x C channels, C*16 B; byte = tile*C*16 + (r>>5)*512 + (p&7)*64 + 4*(p>>3)
//     + 16*((r&15)>>2) + 8*((r>>4)&1) + (r&3), r = storage channel, reference channel = (r&~31) | pi32(r&31).
//  weight slice (16 n x 32 k, 512 B): dword (n&7)*64 + a*16 + ((n>>3)&1)*8 + bb*4 holds storage k 16bb+4a+0..3 of
//     record row n; output storage channel = ipi32(n) within the 32-group.
// Products are formed on f16 = e4m3 * 2^-8 (exact) and accumulated in f32 by v_dot2; the 2^16 is undone afterwards.
#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>
#include "nr_params.h"   // kernarg structs: one definition, shared with tools/build_plan.py

typedef _Float16 h2 __attribute__((ext_vector_type(2)));
typedef _Float16 f16;

__device__ __forceinline__ float dot2(uint32_t a, uint32_t b, float c) {
  return __builtin_amdgcn_fdot2(__builtin_bit_cast(h2, a), __builtin_bit_cast(h2, b), c, false);
}
__device__ __forceinline__ uint32_t e4x2(uint32_t u) {   // two e4m3 bytes (low 16 bits) -> half2 of value * 2^-8
  return ((u & 0x7Fu) << 7) | ((u & 0x80u) << 8) | ((u & 0x7F00u) << 15) | ((u & 0x8000u) << 16);
}
__device__ __forceinline__ uint2 cvt4(uint32_t d) { return make_uint2(e4x2(d & 0xFFFFu), e4x2(d >> 16)); }
__device__ __forceinline__ float q8(float x) {   // f32 -> nearest e4m3 (RNE, saturating)
#ifdef Q8FAST   // f16 + 3-bit mantissa only (2 int ops): keeps the look (CPU 45.3 dB to NVIDIA), no clamp (only swin needs it), no subnormal grid / ties-even
  { const uint32_t q_ = ((uint32_t)__builtin_bit_cast(unsigned short, (_Float16)x) + 0x40u) & 0xFF80u; return (float)__builtin_bit_cast(_Float16, (unsigned short)q_); }
#endif
  x = x == x ? x : 0.f;   // NaN publishes as +0 (OpenDLSS-NR numerics.md), not as the saturated 448
  float a = fminf(fabsf(x), 448.f);
  uint32_t b = __float_as_uint(a);
  float rn = __uint_as_float((b + 0x7FFFFu + ((b >> 20) & 1u)) & 0xFFF00000u);
  float rs = __fsub_rn(__fadd_rn(a, 24576.f), 24576.f);
  return copysignf(a < 0.015625f ? rs : rn, x);
}
__device__ __forceinline__ uint32_t to_e4m3(float v) {   // v on the e4m3 grid
#ifdef Q8BYTE   // f16 bits >> 7 - 64 (sat): tiny values below 2^-6 flush to 0
  { const uint32_t h_ = __builtin_bit_cast(unsigned short, (_Float16)v); const uint32_t a_ = min(h_ & 0x7FFFu, 0x5F00u) >> 7;
    return (a_ > 64u ? a_ - 64u : 0u) | ((h_ >> 8) & 0x80u); }
#endif
#ifdef Q8FAST   // the rounding no longer clamps; a byte store must
  v = copysignf(fminf(fabsf(v), 448.f), v);
#endif
  float a = fabsf(v);
  uint32_t b = a < 0.015625f ? (uint32_t)(a * 512.f) : ((__float_as_uint(a) >> 20) - 960u);
  return b | (v < 0.f ? 0x80u : 0u);
}
__device__ __forceinline__ uint32_t e4m3_byte(float x) {   // nearest e4m3 byte of x; a negative value that rounds to 0 gives 0x80, -0.0 gives 0x00 (as the translated epilogues)
#ifdef Q8BYTE   // f16 bits >> 7 - 64 (sat): tiny values below 2^-6 flush to 0
  { const uint32_t h_ = ((uint32_t)__builtin_bit_cast(unsigned short, (_Float16)x) + 0x40u) & 0xFF80u; const uint32_t a_ = min(h_ & 0x7FFFu, 0x5F00u) >> 7;
    return (a_ > 64u ? a_ - 64u : 0u) | ((h_ >> 8) & 0x80u); }
#endif
  if (x != x) return 0u;   // NaN -> +0
  const uint32_t s = x < 0.f ? 0x80u : 0u;
  const float a = fminf(fabsf(x), 448.f);
  if (a < 0.015625f) return s | (uint32_t)__fsub_rn(__fadd_rn(a * 512.f, 12582912.f), 12582912.f);
  uint32_t b = __float_as_uint(a);
  b = (b + 0x7FFFFu + ((b >> 20) & 1u)) & 0xFFF00000u;
  return s | ((b >> 20) - 960u);
}
__device__ __forceinline__ float store8(float v) { return q8((float)(f16)v); }   // f32 -> f16 -> e4m3
__device__ __forceinline__ float e4f(uint32_t b) {   // one e4m3 byte -> float
  return (float)__builtin_bit_cast(h2, e4x2(b & 0xFFu)).x * 256.f;
}
__device__ __forceinline__ int pi32(int c) { return (c & 1) | (((c >> 2) & 1) << 1) | (((c >> 3) & 1) << 2) | (((c >> 1) & 1) << 3) | (c & 16); }
__device__ __forceinline__ int ipi32(int m) { return (m & 1) | (((m >> 3) & 1) << 1) | (((m >> 1) & 1) << 2) | (((m >> 2) & 1) << 3) | (m & 16); }
__device__ __forceinline__ int tok_off(int p) { return (p & 7) * 64 + 4 * (p >> 3); }          // token p of a tile
__device__ __forceinline__ int ch_off(int r) { return (r >> 5) * 512 + 16 * ((r & 15) >> 2) + 8 * ((r >> 4) & 1); }  // r % 4 == 0

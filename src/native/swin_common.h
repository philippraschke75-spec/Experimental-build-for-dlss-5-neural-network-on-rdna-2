// Helpers shared with native-mid/swinvar.hip (copied verbatim from there: VarParams, q8, e4m3 packing, put16/stage).
#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>
#include "nr_params.h"   // kernarg structs: one definition, shared with tools/build_plan.py

typedef _Float16 h2 __attribute__((ext_vector_type(2)));


__device__ __forceinline__ float q8(float x) {
#ifdef Q8FAST   // f16 + 3-bit mantissa only (2 int ops): keeps the look (CPU 45.3 dB to NVIDIA), +-448 clamp kept (swin exceeds it), no subnormal grid / ties-even
  { const uint32_t q_ = ((uint32_t)__builtin_bit_cast(unsigned short, (_Float16)__builtin_amdgcn_fmed3f(x, -448.f, 448.f)) + 0x40u) & 0xFF80u; return (float)__builtin_bit_cast(_Float16, (unsigned short)q_); }
#endif
  x = x == x ? x : 0.f;   // NaN publishes as +0 (OpenDLSS-NR numerics.md), not as the saturated 448
  float a = fminf(fabsf(x), 448.f);
  uint32_t b = __float_as_uint(a);
  float rn = __uint_as_float((b + 0x7FFFFu + ((b >> 20) & 1u)) & 0xFFF00000u);
  float rs = __fsub_rn(__fadd_rn(a, 24576.f), 24576.f);
  return copysignf(a < 0.015625f ? rs : rn, x);
}
__device__ __forceinline__ float f16r(float x) { return (float)(_Float16)x; }
__device__ __forceinline__ float dot2(uint32_t a, uint32_t b, float c) {
  return __builtin_amdgcn_fdot2(__builtin_bit_cast(h2, a), __builtin_bit_cast(h2, b), c, false);
}
__device__ __forceinline__ uint32_t pk(float a, float b) {
  h2 r; r.x = (_Float16)a; r.y = (_Float16)b; return __builtin_bit_cast(uint32_t, r);
}
__device__ __forceinline__ float lo(uint32_t u) { return (float)__builtin_bit_cast(h2, u).x; }
__device__ __forceinline__ float hi(uint32_t u) { return (float)__builtin_bit_cast(h2, u).y; }
// two e4m3 bytes (low 16 bits of u) -> half2 holding value * 2^-8 (exact, incl. subnormals)
__device__ __forceinline__ uint32_t e4x2(uint32_t u) {
  return ((u & 0x7Fu) << 7) | ((u & 0x80u) << 8) | ((u & 0x7F00u) << 15) | ((u & 0x8000u) << 16);
}
__device__ __forceinline__ uint32_t x256(uint32_t v) {
  return __builtin_bit_cast(uint32_t, __builtin_bit_cast(h2, v) * (h2){(_Float16)256.f, (_Float16)256.f});
}
__device__ __forceinline__ uint32_t to_e4m3(float v) {  // v already on the e4m3 grid
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
__device__ __forceinline__ float act(float x) {
  float t = fminf(fmaxf(x, -4.f), 4.f);
  return __fmul_rn(x, fmaf(t, fmaf(fabsf(t), -0.055908203125f, 0.447265625f), 0.89453125f));
}
__device__ __forceinline__ float norm_scale(const float* v) {  // from native-swin32
  float t[4][2] = {};
#pragma unroll
  for (int g = 0; g < 2; g++)
#pragma unroll
    for (int f = 0; f < 2; f++)
#pragma unroll
      for (int c = 0; c < 4; c++)
#pragma unroll
        for (int h = 0; h < 2; h++) { float x = v[16 * f + 8 * g + 2 * c + h]; t[c][h] = fmaf(x, x, t[c][h]); }
  _Float16 h0 = (_Float16)((t[0][0] + t[1][0]) + (t[2][0] + t[3][0]));
  _Float16 h1 = (_Float16)((t[0][1] + t[1][1]) + (t[2][1] + t[3][1]));
  _Float16 s = h0 + h1;
  const _Float16 floor_ = (_Float16)0.000062f;
  if (s < floor_) s = floor_;
  return __fdiv_rn(1.f, (float)sqrt((double)(float)s));
}
__device__ __forceinline__ int pi32(int c) {   // memory channel -> reference channel (within 32)
  return (c & 1) | (((c >> 2) & 1) << 1) | (((c >> 3) & 1) << 2) | (((c >> 1) & 1) << 3) | (c & 16);
}
__device__ __forceinline__ int ipi32(int m) {  // reference channel -> memory channel
  return (m & 1) | (((m >> 3) & 1) << 1) | (((m >> 1) & 1) << 2) | (((m >> 2) & 1) << 3) | (m & 16);
}
template <int N>
__device__ __forceinline__ void ldw(const uint32_t* p, uint32_t* o) {   // N consecutive LDS words (N = 1, 2, 4, 8)
  if constexpr (N == 1) { o[0] = p[0]; }
  else if constexpr (N == 2) { uint2 u = *(const uint2*)p; o[0] = u.x; o[1] = u.y; }
  else {
#pragma unroll
    for (int i = 0; i < N; i += 4) { uint4 u = *(const uint4*)(p + i); o[i] = u.x; o[i + 1] = u.y; o[i + 2] = u.z; o[i + 3] = u.w; }
  }
}

// one 16-B piece (q = 0..31 within a 512-B tile) -> dst[kq + 4m][ncol + nh*8 + nl] as half2 k-pairs
__device__ __forceinline__ void put16(uint32_t* dst, int ldn, uint4 u, int q, int ncol) {
  const int nl = q >> 2, kq = q & 3;
  uint32_t w0[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int nh = 0; nh < 2; nh++)
#pragma unroll
    for (int m = 0; m < 4; m++) {
      uint32_t word = w0[nh * 2 + (m >> 1)];
      uint32_t v = (m & 1) ? (word >> 16) : word;
      dst[(kq + 4 * m) * ldn + ncol + nh * 8 + nl] = e4x2(v & 0xFFFFu);
    }
}
// Stage `ntiles` tiles; tile i comes from rec byte offset off(i), lands at column ncol(i), kp row kpo(i).
template <typename F>
__device__ __forceinline__ void stage(uint32_t* dst, int ldn, const uint8_t* rec, int ntiles, F where) {
  for (int i = threadIdx.x; i < ntiles * 32; i += 256) {
    int tile = i >> 5, q = i & 31, off, ncol, kpo;
    where(tile, off, ncol, kpo);
    put16(dst + kpo * ldn, ldn, *(const uint4*)(rec + off + q * 16), q, ncol);
  }
}

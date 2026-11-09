// Native gfx1030 kernels for the C=512 split-Swin stage (and, through gemm64, the ViT GEMMs).
// Drop-ins via overrides.txt (rdna2/runtime/nr_override.h). Layouts: native-mid/RESULTS.md.
//  activation: e4m3, 4x4-pixel tiles of C*16 B; byte = tile*C*16 + (r>>5)*512 + (p&7)*64 + 4*(p>>3) + 16*((r&15)>>2)
//     + 8*((r>>4)&1) + (r&3), r = storage channel, reference channel = (r&~31) | pi32(r&31).
//  A storage dword (r = 4t..4t+3) holds reference k {b, b+1, b+8, b+9}; the weight record dword
//     (n&7)*64 + a*16 + ((n>>3)&1)*8 + bb*4 of a 16x32 slice holds the same k set in the same byte order, so
//     activations and weights pair dword by dword. Slices: ((nt/INNER)*(K/32) + kg)*INNER + nt%INNER.
//  planes (stage in/out): [C/16][H][W][16] e4m3, same storage channel order.
// Products are formed on f16 = e4m3 * 2^-8 (exact) and accumulated in f32 by v_dot2; 2^16 is undone in the epilogue.
#include <hip/hip_runtime.h>
#include <stdint.h>
#include "nr_params.h"   // kernarg structs: one definition, shared with tools/build_plan.py

typedef _Float16 h2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ float dot2(uint32_t a, uint32_t b, float c) {
  return __builtin_amdgcn_fdot2(__builtin_bit_cast(h2, a), __builtin_bit_cast(h2, b), c, false);
}
__device__ __forceinline__ uint32_t e4x2(uint32_t u) {   // two e4m3 bytes (low 16 bits) -> half2 of value * 2^-8
  return ((u & 0x7Fu) << 7) | ((u & 0x80u) << 8) | ((u & 0x7F00u) << 15) | ((u & 0x8000u) << 16);
}
__device__ __forceinline__ uint2 cvt4(uint32_t d) { return make_uint2(e4x2(d & 0xFFFFu), e4x2(d >> 16)); }
__device__ __forceinline__ float q8(float x) {   // f32 -> nearest e4m3 (RNE, saturating), from native-mid
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
__device__ __forceinline__ float store8(float v) { return q8((float)(_Float16)v); }   // f32 -> f16 -> e4m3
__device__ __forceinline__ float e4f(uint32_t b) {   // one e4m3 byte -> float
  return (float)__builtin_bit_cast(h2, e4x2(b & 0xFFu)).x * 256.f;
}
__device__ __forceinline__ int pi32(int c) { return (c & 1) | (((c >> 2) & 1) << 1) | (((c >> 3) & 1) << 2) | (((c >> 1) & 1) << 3) | (c & 16); }
__device__ __forceinline__ int ipi32(int m) { return (m & 1) | (((m >> 3) & 1) << 1) | (((m >> 1) & 1) << 2) | (((m >> 2) & 1) << 3) | (m & 16); }
__device__ __forceinline__ int tok_off(int p) { return (p & 7) * 64 + 4 * (p >> 3); }          // token p of a tile
__device__ __forceinline__ int ch_off(int r) { return (r >> 5) * 512 + 16 * ((r & 15) >> 2) + 8 * ((r >> 4) & 1); }  // r % 4 == 0

// 64 tokens (4 tiles from tile0, tiles >= T read as zero) x 64 outputs (storage rows n0..n0+63) over K.
// Thread (ty = tid>>4, tx = tid&15) owns tokens 4ty..4ty+3 and storage rows n0+4tx..+3: acc[i][e].
// LDS: 2 buffers x (A[16 kd][64 tok] + W[16 kd][64 n]) uint2 = 32 KB.
// Per-thread weight staging: 16 B at wsrc (+ c * wstep for chunk c); dword e lands at W[wk[e]][wn[e]].
struct WPiece { const uint8_t* src; size_t step; int wn[4], wk[4]; };

// standard record: 16x32 slices, ((nt/INNER)*(K/32) + kg)*INNER + nt%INNER
template <int K, int INNER>
__device__ __forceinline__ WPiece wpiece_std(const uint8_t* Wr, int n0, int ntmax = 1 << 30) {
  const int tid = threadIdx.x, q = tid & 31, ntl = tid >> 5 & 3, kgl = tid >> 7, nt = (n0 >> 4) + ntl;
  const int ns = nt < ntmax ? nt : ntmax;   // rows past ntmax read a valid slice; their outputs are discarded
  WPiece w;
  w.src = Wr + (size_t)(((ns / INNER) * (K / 32) + kgl) * INNER + ns % INNER) * 512 + q * 16;
  w.step = (size_t)2 * INNER * 512;   // two k-groups further
#pragma unroll
  for (int e = 0; e < 4; e++) {
    const int n = nt * 16 + (e >> 1) * 8 + (q >> 2);
    w.wn[e] = ((n & ~31) | ipi32(n & 31)) - n0;
    w.wk[e] = kgl * 8 + (e & 1) + 2 * (q & 3);
  }
  return w;
}

__device__ __forceinline__ int lin_tile(int tile0, int T) { const int t = tile0 + (threadIdx.x >> 6); return t < T ? t : -1; }

// atile: the tile of this thread's A piece (token group tid>>6), -1 = zero tokens
// 16 kd steps of the 4x4 micro tile from LDS: As[kd][64 tok], Ws[kd][64 n] (uint2 = 4 halves)
__device__ __forceinline__ void mma16(const uint2* As, const uint2* Ws, float acc[4][4]) {
  const int ty = threadIdx.x >> 4, tx = threadIdx.x & 15;
#pragma unroll 4
  for (int kd = 0; kd < 16; kd++) {
    const uint4 a01 = *(const uint4*)(As + kd * 64 + ty * 4), a23 = *(const uint4*)(As + kd * 64 + ty * 4 + 2);
    const uint4 w01 = *(const uint4*)(Ws + kd * 64 + tx * 4), w23 = *(const uint4*)(Ws + kd * 64 + tx * 4 + 2);
    const uint32_t A[4][2] = {{a01.x, a01.y}, {a01.z, a01.w}, {a23.x, a23.y}, {a23.z, a23.w}};
    const uint32_t B[4][2] = {{w01.x, w01.y}, {w01.z, w01.w}, {w23.x, w23.y}, {w23.z, w23.w}};
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
      for (int e = 0; e < 4; e++) acc[i][e] = dot2(A[i][1], B[e][1], dot2(A[i][0], B[e][0], acc[i][e]));
  }
}

// X in planes [C/16][H][W][16] instead of tiles (first C=512 block): the four dwords of a piece are loaded one by one.
struct Planes { const uint8_t* p; int H, W; };

template <int K>
__device__ __forceinline__ void gemm64(const uint8_t* __restrict__ X, int atile, const WPiece& wp,
                                       float acc[4][4], uint2* lds, Planes pl = Planes{nullptr, 0, 0}) {
  constexpr int NC = K / 64;
  const int tid = threadIdx.x;
  size_t poff[4] = {0, 0, 0, 0};
  if (pl.p && atile >= 0) {
    const int q = tid & 31, sa = (tid >> 5) & 1, TW = pl.W >> 2;
#pragma unroll
    for (int e = 0; e < 4; e++) {
      const int p = (q >> 2) + 8 * (e & 1), ch = sa * 32 + (e >> 1) * 16 + (q & 3) * 4;
      const int row = (atile / TW) * 4 + (p >> 2), col = (atile % TW) * 4 + (p & 3);
      poff[e] = (size_t)(ch >> 4) * pl.H * pl.W * 16 + (size_t)(row * pl.W + col) * 16 + (ch & 15);
    }
  }
  const size_t pstep = (size_t)4 * pl.H * pl.W * 16;   // 64 channels = 4 planes
  auto loadA = [&](int c) -> uint4 {
    if (atile < 0) return make_uint4(0, 0, 0, 0);
    if (pl.p) {
      const uint8_t* b = pl.p + c * pstep;
      return make_uint4(*(const uint32_t*)(b + poff[0]), *(const uint32_t*)(b + poff[1]),
                        *(const uint32_t*)(b + poff[2]), *(const uint32_t*)(b + poff[3]));
    }
    return *(const uint4*)(X + (size_t)atile * (K * 16) + ((tid >> 5) & 1) * 512 + (tid & 31) * 16 + (size_t)c * 1024);
  };
  // A piece: tile tt, slice sa, 16 B q
  const int tt = tid >> 6, sa = (tid >> 5) & 1, q = tid & 31;
  const uint8_t* wsrc = wp.src;
  const size_t wstep = wp.step;
  int am[4], ak[4], wn[4], wk[4];
#pragma unroll
  for (int e = 0; e < 4; e++) {
    am[e] = tt * 16 + (q >> 2) + 8 * (e & 1);
    ak[e] = sa * 8 + 2 * (q & 3) + (e >> 1);
    wn[e] = wp.wn[e]; wk[e] = wp.wk[e];
  }
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int e = 0; e < 4; e++) acc[i][e] = 0.f;

  uint4 ra = loadA(0);
  uint4 rw = *(const uint4*)wsrc;
  for (int c = 0; c < NC; c++) {
    uint2* As = lds + (c & 1) * 2048;
    uint2* Ws = As + 1024;
    const uint32_t av[4] = {ra.x, ra.y, ra.z, ra.w}, wv[4] = {rw.x, rw.y, rw.z, rw.w};
#pragma unroll
    for (int e = 0; e < 4; e++) { As[ak[e] * 64 + am[e]] = cvt4(av[e]); Ws[wk[e] * 64 + wn[e]] = cvt4(wv[e]); }
    __syncthreads();
    if (c + 1 < NC) {
      ra = loadA(c + 1);
      rw = *(const uint4*)(wsrc + (size_t)(c + 1) * wstep);
    }
    mma16(As, Ws, acc);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// ViT (blocks 31-38): tokens in 16-token tiles; the translated kernels have no token count, they cover gridDim.x * 64.

__device__ __forceinline__ float vit_act(float x) {
  const float t = fminf(fmaxf(x, -4.f), 4.f);
  return __fmul_rn(x, fmaf(t, fmaf(fabsf(t), -0.055908203125f, 0.447265625f), 0.89453125f));
}

// grid (=, 64): 4 tiles x 64 of the 4096 hidden channels
extern "C" __global__ void __launch_bounds__(256) k_vit_expand_n(ExpandParams P) {
  __shared__ uint2 lds[4096];
  const int tile0 = blockIdx.x * 4, n0 = blockIdx.y * 64, T = gridDim.x * 4;
  if (n0 >= 4096) return;
  float acc[4][4];
  gemm64<1024>(P.x, lin_tile(tile0, T), wpiece_std<1024, 256>(P.w, n0), acc, lds);
  const int ty = threadIdx.x >> 4, r = n0 + (threadIdx.x & 15) * 4;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int m = ty * 4 + i, tile = tile0 + (m >> 4);
    uint32_t o = 0;
#pragma unroll
    for (int e = 0; e < 4; e++) o |= to_e4m3(store8(vit_act(acc[i][e] * 65536.f))) << (8 * e);
    *(uint32_t*)(P.out + (size_t)tile * 65536 + ch_off(r) + tok_off(m & 15)) = o;
  }
}

// grid (=, 16): 4 tiles x 64 of the 1024 outputs; out = e4m3(f16(W x + gain * skip))
template <int K>
__device__ __forceinline__ void vit_contract(const ContractParams& P, uint2* lds) {
  const int tile0 = blockIdx.x * 4, n0 = blockIdx.y * 64, T = gridDim.x * 4;
  float acc[4][4];
  gemm64<K>(P.x, lin_tile(tile0, T), wpiece_std<K, 64>(P.w, n0), acc, lds);
  const int ty = threadIdx.x >> 4, r = n0 + (threadIdx.x & 15) * 4;
  const _Float16* gain = (const _Float16*)(P.w + 1024 * K);
  float g[4];
#pragma unroll
  for (int e = 0; e < 4; e++) g[e] = (float)gain[(r & ~31) | pi32((r + e) & 31)];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int m = ty * 4 + i, tile = tile0 + (m >> 4);
    const size_t off = (size_t)tile * 16384 + ch_off(r) + tok_off(m & 15);
    const uint32_t sd = *(const uint32_t*)(P.skip + off);
    uint32_t o = 0;
#pragma unroll
    for (int e = 0; e < 4; e++)
      o |= to_e4m3(store8(__fadd_rn(acc[i][e] * 65536.f, __fmul_rn(e4f(sd >> (8 * e)), g[e])))) << (8 * e);
    *(uint32_t*)(P.out + off) = o;
  }
}
extern "C" __global__ void __launch_bounds__(256) k_vit_contract_n(ContractParams P) {
  __shared__ uint2 lds[4096];
  if (blockIdx.y >= 16) return;
  if (P.K == 4096) vit_contract<4096>(P, lds);
  else if (P.K == 1024) vit_contract<1024>(P, lds);
}

// k_qkv2 (VitQKV): record = header f32[32] temperatures (128 B), then q,k,v [1024][1024] interleaved in 1024-B tiles
// of 32x32: tile = (n>>5) | ((k>>5)<<5) at (3*tile + which)*1024, byte bits: k0,k3,k4,n3,k1,k2,n0,n1,n2,n4.

__device__ __forceinline__ WPiece wpiece_qkv3(const uint8_t* rec, int which, int n0) {
  const int tid = threadIdx.x, s = tid >> 6, rtl = s & 1, ktl = s >> 1, u = tid & 63;
  const int tile = ((n0 >> 5) + rtl) | (ktl << 5);
  WPiece w;
  w.src = rec + 128 + (size_t)(3 * tile + which) * 1024 + u * 16;
  w.step = (size_t)3 * 64 * 1024;   // k-tile + 2
#pragma unroll
  for (int e = 0; e < 4; e++) {
    const int row = (u >> 2 & 1) | (u >> 3 & 1) << 1 | (u >> 4 & 1) << 2 | (e >> 1) << 3 | (u >> 5 & 1) << 4;
    w.wn[e] = rtl * 32 + ipi32(row);
    w.wk[e] = ktl * 8 + (e & 1) + 2 * ((u & 1) | (u >> 1 & 1) << 1);
  }
  return w;
}

typedef _Float16 f16;
__device__ __forceinline__ f16 xorh(f16 v, int m) { return (f16)__shfl_xor((float)v, m, 32); }

// grid (=, 48): which = y / 16, 64 of its 1024 outputs (two heads)
extern "C" __global__ void __launch_bounds__(256) k_vit_qkv_n(QkvParams P) {
  __shared__ uint2 lds[4096];
  if (blockIdx.y >= 48) return;
  const int which = blockIdx.y >> 4, n0 = (blockIdx.y & 15) * 64, tile0 = blockIdx.x * 4;
  float acc[4][4];
  gemm64<1024>(P.x, lin_tile(tile0, gridDim.x * 4), wpiece_qkv3(P.w, which, n0), acc, lds);
  const int ty = threadIdx.x >> 4, r = n0 + (threadIdx.x & 15) * 4;
  const f16 temp = (f16)((const float*)P.w)[r >> 5];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int m = ty * 4 + i, tile = tile0 + (m >> 4);
    float y[4];
    if (which == 2) {
#pragma unroll
      for (int e = 0; e < 4; e++) y[e] = store8(acc[i][e] * 65536.f);
    } else {
      f16 x[4], s1[4];
#pragma unroll
      for (int e = 0; e < 4; e++) { x[e] = (f16)(acc[i][e] * 65536.f); s1[e] = x[e] * x[e]; }
#pragma unroll
      for (int e = 0; e < 4; e++) s1[e] = s1[e] + xorh(s1[e], 4);   // d + (d+16)
      f16 ta = s1[0] + s1[2], tb = s1[1] + s1[3];                     // d + (d+8)
      ta = ta + xorh(ta, 2); tb = tb + xorh(tb, 2);                   // d + (d+4)
      ta = ta + xorh(ta, 1); tb = tb + xorh(tb, 1);                   // d + (d+2)
      f16 tot = ta + tb;                                              // 0 + 1
      if (tot < (f16)0.000062f) tot = (f16)0.000062f;
      const f16 inv = (f16)(1.f / sqrtf((float)tot));
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const f16 v = which == 0 ? (f16)((f16)(x[e] * (f16)5.65625f) * inv) * temp : x[e] * inv;
        y[e] = q8((float)v);
      }
    }
    uint32_t o = 0;
#pragma unroll
    for (int e = 0; e < 4; e++) o |= to_e4m3(y[e]) << (8 * e);
    *(uint32_t*)(P.out[which] + (size_t)tile * 16384 + ch_off(r) + tok_off(m & 15)) = o;
  }
}

// ---------------------------------------------------------------------------------------------------------------
// C=512 stage GEMM (perf-c512): 64 tokens x 16*NS storage rows; thread (ty, tx) owns tokens 4ty..4ty+3 and rows
// NS*tx..NS*tx+NS-1. One chunk = one 32-channel k-group (8 kd): A[8][64] + W[8][16NS] uint2, double-buffered
// (NS=4: 16 KB, NS=6: 20 KB) - half (NS=4) / 0.6x (NS=6) of gemm64's 32 KB, so 2-3 workgroups fit per CU.
// The per-output accumulation sequence is gemm64's exactly (k-groups in order, kd = 2*((r&15)>>2) + ((r>>4)&1)
// within a group, dword .x then .y), so the results are bit-identical to the gemm64 kernels.
// Per chunk a thread loads 8 B of A (tokens m, m+8 at one kd) and, for tid < 32*NS, one 16 B weight piece.
struct WP8 { const uint8_t* src; size_t step; int wn[4], wk[4]; bool on; };   // wk < 8; step = bytes per k-group

template <int K, int INNER, int NS>
__device__ __forceinline__ WP8 wp8_std(const uint8_t* Wr, int n0) {
  const int tid = threadIdx.x, q = tid & 31, piece = tid >> 5;
  WP8 w; w.on = piece < NS;
  const int nt = (n0 >> 4) + (w.on ? piece : 0);
  w.src = Wr + (size_t)((nt / INNER) * (K / 32) * INNER + nt % INNER) * 512 + q * 16;
  w.step = (size_t)INNER * 512;
#pragma unroll
  for (int e = 0; e < 4; e++) {
    const int n = nt * 16 + (e >> 1) * 8 + (q >> 2);
    w.wn[e] = ((n & ~31) | ipi32(n & 31)) - n0;
    w.wk[e] = (e & 1) + 2 * (q & 3);
  }
  return w;
}

#ifndef MMA8_UNROLL
#define MMA8_UNROLL 2
#endif
template <int NS>
__device__ __forceinline__ void mma8(const uint2* As, const uint2* Ws, float acc[4][NS]) {
  const int ty = threadIdx.x >> 4, tx = threadIdx.x & 15;
#pragma unroll MMA8_UNROLL
  for (int kd = 0; kd < 8; kd++) {
    const uint4 a01 = *(const uint4*)(As + kd * 64 + ty * 4), a23 = *(const uint4*)(As + kd * 64 + ty * 4 + 2);
    const uint32_t A[4][2] = {{a01.x, a01.y}, {a01.z, a01.w}, {a23.x, a23.y}, {a23.z, a23.w}};
    uint32_t B[NS][2];
#pragma unroll
    for (int e2 = 0; e2 < NS / 2; e2++) {
      const uint4 w = *(const uint4*)(Ws + kd * (16 * NS) + tx * NS + 2 * e2);
      B[2 * e2][0] = w.x; B[2 * e2][1] = w.y; B[2 * e2 + 1][0] = w.z; B[2 * e2 + 1][1] = w.w;
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
      for (int e = 0; e < NS; e++) acc[i][e] = dot2(A[i][1], B[e][1], dot2(A[i][0], B[e][0], acc[i][e]));
  }
}

template <int K, int NS>
__device__ __forceinline__ void gemm_k32(const uint8_t* __restrict__ X, int atile, const WP8& wp, float acc[4][NS],
                                         uint2* lds, Planes pl = Planes{nullptr, 0, 0}) {
  constexpr int NC = K / 32, AB = 8 * 64, WB = 8 * 16 * NS, BUF = AB + WB;
  const int tid = threadIdx.x, q = tid & 31, tt = tid >> 6, half = (tid >> 5) & 1;
  const int am = tt * 16 + (q >> 2), ak = 2 * (q & 3) + half;   // dwords: tokens am, am+8 at kd ak
  const uint8_t* asrc = nullptr; size_t poff[2] = {0, 0}, pstep = 0;
  if (atile >= 0) {
    if (pl.p) {   // planes: dword u = channel 32kg + 16half + 4(q&3), token p = (q>>2) + 8u
      const int TW = pl.W >> 2, ch = half * 16 + (q & 3) * 4;
#pragma unroll
      for (int u = 0; u < 2; u++) {
        const int p = (q >> 2) + 8 * u, row = (atile / TW) * 4 + (p >> 2), col = (atile % TW) * 4 + (p & 3);
        poff[u] = (size_t)(ch >> 4) * pl.H * pl.W * 16 + (size_t)(row * pl.W + col) * 16 + (ch & 15);
      }
      pstep = (size_t)2 * pl.H * pl.W * 16;   // 32 channels = 2 planes
      asrc = pl.p;
    } else asrc = X + (size_t)atile * (K * 16) + q * 16 + half * 8;
  }
  auto loadA = [&](int c) -> uint2 {
    if (!asrc) return make_uint2(0, 0);
    if (pl.p) { const uint8_t* b = asrc + c * pstep; return make_uint2(*(const uint32_t*)(b + poff[0]), *(const uint32_t*)(b + poff[1])); }
    return *(const uint2*)(asrc + (size_t)c * 512);
  };
  auto loadW = [&](int c) -> uint4 { return wp.on ? *(const uint4*)(wp.src + (size_t)c * wp.step) : make_uint4(0, 0, 0, 0); };
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int e = 0; e < NS; e++) acc[i][e] = 0.f;
  uint2 ra = loadA(0);
  uint4 rw = loadW(0);
  for (int c = 0; c < NC; c++) {
    uint2* As = lds + (c & 1) * BUF;
    uint2* Ws = As + AB;
    As[ak * 64 + am] = cvt4(ra.x); As[ak * 64 + am + 8] = cvt4(ra.y);
    if (wp.on) {
      const uint32_t wv[4] = {rw.x, rw.y, rw.z, rw.w};
#pragma unroll
      for (int e = 0; e < 4; e++) Ws[wp.wk[e] * (16 * NS) + wp.wn[e]] = cvt4(wv[e]);
    }
    __syncthreads();
    if (c + 1 < NC) { ra = loadA(c + 1); rw = loadW(c + 1); }
    mma8<NS>(As, Ws, acc);
  }
}

#ifndef CONV2_WAVES
#define CONV2_WAVES 4
#endif
#ifndef ATTN_WAVES
#define ATTN_WAVES 4
#endif
#ifndef FFWD_WAVES
#define FFWD_WAVES 4
#endif
#ifndef CONV2_ATTR
#define CONV2_ATTR
#endif
#ifndef ATTN_ATTR
#define ATTN_ATTR
#endif
#ifndef FFWD_ATTR
#define FFWD_ATTR
#endif
#ifndef FFWD_LDS
#define FFWD_LDS 3072
#endif

// ---------------------------------------------------------------------------------------------------------------
// k_conv_res2(SplitFfwdProj / SplitProj): out = e4m3(f16(W x + gain * res)), C = 512.

// grid (ceil(T/4), 8): 4 tiles x 64 storage channels per workgroup; LDS 16 KB
extern "C" __global__ void __launch_bounds__(256, CONV2_WAVES) CONV2_ATTR k_conv2_n(Conv2Params P) {
  __shared__ uint2 lds[2048];
  const int tile0 = blockIdx.x * 4, n0 = blockIdx.y * 64;
  if (n0 >= 512) return;
  float acc[4][4];
  gemm_k32<512, 4>(P.x, lin_tile(tile0, P.T), wp8_std<512, 32, 4>(P.w, n0), acc, lds);
  const int ty = threadIdx.x >> 4, r = n0 + (threadIdx.x & 15) * 4, WT = P.W >> 2;
  const _Float16* gain = (const _Float16*)(P.w + 512 * 512);
  float g[4];
#pragma unroll
  for (int e = 0; e < 4; e++) g[e] = (float)gain[(r & ~31) | pi32((r + e) & 31)];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int m = ty * 4 + i, tile = tile0 + (m >> 4), p = m & 15;
    if (tile >= P.T) continue;
    const size_t off = (size_t)tile * 8192 + ch_off(r) + tok_off(p);
    const int row = (tile / WT) * 4 + (p >> 2), col = (tile % WT) * 4 + (p & 3);
    const size_t poff = (size_t)(r >> 4) * P.H * P.W * 16 + (size_t)(row * P.W + col) * 16 + (r & 15);
    uint32_t rd = 0;
    if (P.res) rd = *(const uint32_t*)(P.res + off);
    else if (P.resp) rd = *(const uint32_t*)(P.resp + poff);
    uint32_t o = 0;
#pragma unroll
    for (int e = 0; e < 4; e++) {
      const float v = __fadd_rn(acc[i][e] * 65536.f, __fmul_rn(e4f(rd >> (8 * e)), g[e]));
      o |= to_e4m3(store8(v)) << (8 * e);
    }
    *(uint32_t*)(P.out + off) = o;
    if (P.out2) *(uint32_t*)(P.out2 + poff) = o;
  }
}

// ---------------------------------------------------------------------------------------------------------------
// k_qkv_attn2 (SplitQKVAttn): one 8x8 window (shifted by origin ox, oy in {0,-4}) x one head per workgroup.
// Record: qkv [1536][512] (INNER 96; head h rows h*96 + {q,k,v}*32 + d), attn_pos_bias f16[16][4096] in
// accumulator-fragment order, tail f32[16] per-head query scale. Window token = 16*tile + p (tiles TL, TR, BL, BR);
// tiles outside the map are zero tokens (they act as keys, their outputs are dropped).
// q, k and v come from ONE 64x96 GEMM (A loaded once, no discarded rows); the f16 results go through LDS (Xs) so the
// epilogue runs in the per-head lane layout the shuffle reductions were verified in. LDS, all regions aliased
// into 21248 B (3 workgroups per CU):  [0, 12288) GEMM staging / Xs f16[64][96] / Pr f16[64][66]
//                                      [12288, 20992) GEMM staging / Qs, Ks f16[64][34] / Vs f32[64][33]
//                                      [20992, 21248) Inv f32[64]

__device__ __forceinline__ f16 head_inv_norm(const f16 x[4]) {   // lanes tx..tx^7 hold one head (storage rows)
  f16 s1[4];
#pragma unroll
  for (int e = 0; e < 4; e++) { s1[e] = x[e] * x[e]; s1[e] = s1[e] + xorh(s1[e], 4); }   // d + (d+16)
  f16 ta = s1[0] + s1[2], tb = s1[1] + s1[3];                                            // d + (d+8)
  ta = ta + xorh(ta, 2); tb = tb + xorh(tb, 2);                                          // d + (d+4)
  ta = ta + xorh(ta, 1); tb = tb + xorh(tb, 1);
  f16 tot = ta + tb;
  if (tot < (f16)0.000062f) tot = (f16)0.000062f;
  return (f16)(1.f / sqrtf((float)tot));
}

extern "C" __global__ void __launch_bounds__(256, ATTN_WAVES) ATTN_ATTR k_qkv_attn_n(AttnParams P) {
  __shared__ __align__(16) uint8_t smem[21248];
  uint2* lds = (uint2*)smem;                       // GEMM staging: 2 x (512 + 768) uint2 = 20480 B
  f16* Xs = (f16*)smem;                            // [64][96]
  f16* Pr = (f16*)smem;                            // [64][66], after Xs is consumed
  f16* Qs = (f16*)(smem + 12288); f16* Ks = Qs + 64 * 34;
  float* Vs = (float*)(smem + 12288);              // [64][33], after Qs/Ks are consumed
  float* Inv = (float*)(smem + 20992);
  const int h = blockIdx.z, tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;
  if (h >= 16) return;
  const int TH = P.H >> 2, TW = P.W >> 2;
  auto win_tile = [&](int tt) {
    const int tr = 2 * (int)blockIdx.y + (P.oy >> 2) + (tt >> 1), tc = 2 * (int)blockIdx.x + (P.ox >> 2) + (tt & 1);
    return (tr >= 0 && tr < TH && tc >= 0 && tc < TW) ? tr * TW + tc : -1;
  };
  const int atile = win_tile(tid >> 6);
  float acc[4][6];
  gemm_k32<512, 6>(P.x, atile, wp8_std<512, 96, 6>(P.w, h * 96), acc, lds);
  __syncthreads();   // staging reads done
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int e = 0; e < 6; e++) Xs[(ty * 4 + i) * 96 + tx * 6 + e] = (f16)(acc[i][e] * 65536.f);
  __syncthreads();
#ifdef EXP_ATTN_GEMM_ONLY
  if (atile >= 0) P.out[(size_t)atile * 8192 + tid * 2] = (uint8_t)Xs[tid * 24 + h];
  return;
#endif
  const int rr = (tx & 7) * 4;   // storage row within the head
  float vreg[4][4];
  {
    const f16 scale = (f16)((const float*)(P.w + 1536 * 512 + 16 * 8192))[h];
    const int cb = (tx < 8 ? 0 : 32) + rr;   // q rows for tx < 8, k rows for tx >= 8
#pragma unroll
    for (int i = 0; i < 4; i++) {
      f16 x[4];
#pragma unroll
      for (int e = 0; e < 4; e++) x[e] = Xs[(ty * 4 + i) * 96 + cb + e];
      const f16 inv = head_inv_norm(x);
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const f16 v = x[e] * inv;
        if (tx < 8) Qs[(ty * 4 + i) * 34 + rr + e] = (f16)q8((float)(f16)(v * scale));
        else Ks[(ty * 4 + i) * 34 + rr + e] = (f16)q8((float)v);
      }
    }
    if (tx < 8) {
#pragma unroll
      for (int i = 0; i < 4; i++)
#pragma unroll
        for (int e = 0; e < 4; e++) vreg[i][e] = store8((float)Xs[(ty * 4 + i) * 96 + 64 + rr + e]);
    }
  }
  __syncthreads();   // Qs/Ks visible, Xs consumed
  // logits + exponent: query i = tid>>2, keys j = (tid&3) + 4l
  {
    const int i = tid >> 2;
    uint32_t qv[16];
#pragma unroll
    for (int d = 0; d < 16; d++) qv[d] = *(const uint32_t*)(Qs + i * 34 + 2 * d);
    const _Float16* bias = (const _Float16*)(P.w + 1536 * 512 + h * 8192);
    for (int l = 0; l < 16; l++) {
      const int j = (tid & 3) + 4 * l;
      float s = 0.f;
#pragma unroll
      for (int d = 0; d < 16; d++) s = dot2(qv[d], *(const uint32_t*)(Ks + j * 34 + 2 * d), s);
      const int bi = (4 * (i >> 4) + (j >> 4)) * 256 + 8 * (4 * (i & 7) + ((j & 7) >> 1)) + ((j & 1) | (((i >> 3) & 1) << 1) | (((j >> 3) & 1) << 2));
      const float lg = __fadd_rn(s, (float)bias[bi]);
      f16 y = __builtin_fmaf16((f16)lg, (f16)0.044921875f, (f16)1.30078125f);
      y = y < (f16)1.03125f ? (f16)1.03125f : (y > (f16)1.5693359375f ? (f16)1.5693359375f : y);
      const uint16_t bits = (uint16_t)((__builtin_bit_cast(uint16_t, y) & 0x3FF) << 5);
      Pr[i * 66 + j] = __builtin_bit_cast(f16, bits);
    }
  }
  __syncthreads();   // Pr visible, Qs/Ks consumed
  if (tx < 8) {
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
      for (int e = 0; e < 4; e++) Vs[(ty * 4 + i) * 33 + rr + e] = vreg[i][e];
  }
  if (tid < 64) {   // denominator in the reference order
    const f16* p = Pr + tid * 66;
    float hv[2];
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      float col[8];
#pragma unroll
      for (int c = 0; c < 8; c++)
        col[c] = __fadd_rn(__fadd_rn(__fadd_rn((float)p[8 * hh + c], (float)p[16 + 8 * hh + c]), (float)p[32 + 8 * hh + c]), (float)p[48 + 8 * hh + c]);
      hv[hh] = __fadd_rn(__fadd_rn(col[0] + col[1], col[2] + col[3]), __fadd_rn(col[4] + col[5], col[6] + col[7]));
    }
    Inv[tid] = __fdiv_rn(1.f, fmaxf(__fadd_rn(hv[0], hv[1]), 6.103515625e-05f));
  }
  __syncthreads();
  {   // P = store8(Pr * inv) once per element, in place (the 4 threads of a query each recomputed all 64 before; 3z).
      // store8 is on the e4m3 grid, so the f16 holds it exactly: the P.V loop below sees the same values.
    const int i = tid >> 2;
    const float inv = Inv[i];
#pragma unroll
    for (int l = 0; l < 16; l++) {
      const int j = (tid & 3) + 4 * l;
      Pr[i * 66 + j] = (f16)store8(__fmul_rn((float)Pr[i * 66 + j], inv));
    }
  }
  __syncthreads();
  {
    const int i = tid >> 2, cg = (tid & 3) * 8, tile = win_tile(i >> 4);
    float c8[8] = {};
    for (int j = 0; j < 64; j++) {
      const float pj = (float)Pr[i * 66 + j];
#pragma unroll
      for (int c = 0; c < 8; c++) c8[c] = fmaf(pj, Vs[j * 33 + cg + c], c8[c]);
    }
    if (tile >= 0) {
#pragma unroll
      for (int u = 0; u < 2; u++) {
        uint32_t o = 0;
#pragma unroll
        for (int e = 0; e < 4; e++) o |= to_e4m3(store8(c8[4 * u + e])) << (8 * e);
        *(uint32_t*)(P.out + (size_t)tile * 8192 + ch_off(32 * h + cg + 4 * u) + tok_off(i & 15)) = o;
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// k_ffwd2 (SplitFfwd): group g = blockIdx.y: R = e4m3(f16(A_g x)) [64], then per 64-wide hidden chunk
// Hc = e4m3(f16 act(Q0_g R)), out += Q2_g Hc; out channels 64g..64g+63 = e4m3(f16(out)).
// Record bit scatter (split_swin.py _grouped_ffwd_matrices):
//  A_g [64 r][512 k]: g<<11 | kc<<14 (k>>5) | k0,k3,k4,r0,k1,k2,r1..r5 -> bits 0..10
//  Q0_g [256 j][64 r]: 262144 + g<<14 | r1,r0,r4,j1,r2,r3,j0,j3,j4,j2,j5,j6,j7,r5 -> bits 0..13
//  Q2_g [64 n][256 j]: 393216 + g<<14 | j0,j1,j2,n3,j3,j4,n0,n1,n2,n4,n5,j5,j6,j7 -> bits 0..13
// LDS 24 KB: stage-1 staging (16 KB), then Ws[16][64] | Rs | Hs (8 KB each). The Q0/Q2 chunk loads are issued one
// phase ahead so their latency hides behind the previous mma16.

__device__ __forceinline__ uint32_t pkh(float a, float b) {
  h2 r; r.x = (f16)a; r.y = (f16)b; return __builtin_bit_cast(uint32_t, r);
}
__device__ __forceinline__ float split_act(float xf) {   // cubic SiLU in f16 arithmetic
  const f16 x = (f16)xf;
  const f16 t = x < (f16)-4.f ? (f16)-4.f : (x > (f16)4.f ? (f16)4.f : x);
  const f16 u = __builtin_fmaf16((f16)-0.055908203125f, t < (f16)0.f ? -t : t, (f16)0.447265625f);
  const f16 v = __builtin_fmaf16(t, u, (f16)0.89453125f);
  return (float)(f16)(x * v);
}

extern "C" __global__ void __launch_bounds__(256, FFWD_WAVES) FFWD_ATTR k_ffwd2_n(Ffwd2Params P) {
  __shared__ uint2 lds[FFWD_LDS];
  const int g = blockIdx.y, tid = threadIdx.x, ty = tid >> 4, tx = tid & 15, tile0 = blockIdx.x * 4;
  if (g >= 8) return;
  const uint8_t* rec = P.w;
  // stage 1: A_g (64 rows r, K = 512); one 2 KB k-group piece per 128 threads; W dword e: k4 = e&1, r0 = e>>1
  WP8 wa;
  {
    const int u = tid & 127;
    wa.on = tid < 128;
    wa.src = rec + (g << 11) + u * 16;
    wa.step = (size_t)1 << 14;
#pragma unroll
    for (int e = 0; e < 4; e++) { wa.wn[e] = (e >> 1) | (u >> 2) << 1; wa.wk[e] = (e & 1) + 2 * (u & 3); }
  }
  float acc[4][4];
  gemm_k32<512, 4>(P.x, lin_tile(tile0, P.T), wa, acc, lds, Planes{P.x ? nullptr : P.xp, P.H, P.W});
  uint2* Ws = lds; uint2* Rs = lds + 1024; uint2* Hs = lds + 2048;
  const uint8_t* q0src = rec + 262144 + (g << 14) + ((tid >> 7) << 13) + (tid & 127) * 16;   // + jc << 11
  const uint8_t* q2src = rec + 393216 + (g << 14) + tid * 16;                                 // + jc << 12
  uint4 q0 = *(const uint4*)q0src;
  __syncthreads();   // stage-1 staging reads done before Rs overwrites it
  // R in Q0's dword order: kd = r>>2, lo = (r, r+2), hi = (r+1, r+3); thread holds r = 4tx..4tx+3
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float s[4];
#pragma unroll
    for (int e = 0; e < 4; e++) s[e] = store8(acc[i][e] * 65536.f);
    Rs[tx * 64 + ty * 4 + i] = make_uint2(pkh(s[0], s[2]) , pkh(s[1], s[3]));
  }
  float out[4][4];
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int e = 0; e < 4; e++) out[i][e] = 0.f;
#ifdef EXP_FFWD_STAGE1_ONLY
  for (int jc = 0; jc < 0; jc++) {
#else
  for (int jc = 0; jc < 4; jc++) {
#endif
    // Q0 chunk: 64 j x 64 r. thread: r5 = tid>>7, u = tid&127; dword e: r4 = e&1, j1 = e>>1
    __syncthreads();   // previous users of Ws (last Q2) are done; Rs visible
    {
      const int r5 = tid >> 7, u = tid & 127;
      const uint32_t d[4] = {q0.x, q0.y, q0.z, q0.w};
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const int j = (u >> 2 & 1) | (e >> 1) << 1 | (u >> 5 & 1) << 2 | (u >> 3 & 1) << 3 | (u >> 4 & 1) << 4 | (u >> 6 & 1) << 5;
        const int kd = (u & 1) | (u >> 1 & 1) << 1 | (e & 1) << 2 | r5 << 3;
        Ws[kd * 64 + j] = cvt4(d[e]);
      }
    }
    const uint4 q2 = *(const uint4*)(q2src + (jc << 12));
    __syncthreads();
    float hc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
      for (int e = 0; e < 4; e++) hc[i][e] = 0.f;
    mma16(Rs, Ws, hc);   // hidden j = 4tx+e of this chunk
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float s[4];
#pragma unroll
      for (int e = 0; e < 4; e++) s[e] = q8(split_act(hc[i][e] * 256.f));
      Hs[tx * 64 + ty * 4 + i] = make_uint2(pkh(s[0], s[1]), pkh(s[2], s[3]));   // Q2 order: j..j+3
    }
    __syncthreads();   // Ws (Q0) reads and Hs writes done
    // Q2 chunk: 64 n x 64 j; dword e: j2 = e&1, n3 = e>>1; row stored at its storage position
    {
      const int u = tid;
      const uint32_t d[4] = {q2.x, q2.y, q2.z, q2.w};
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const int n = (u >> 2 & 7) | (e >> 1) << 3 | (u >> 5 & 3) << 4;
        const int kd = (e & 1) | (u & 3) << 1 | (u >> 7) << 3;
        Ws[kd * 64 + ((n & 32) | ipi32(n & 31))] = cvt4(d[e]);
      }
    }
    if (jc + 1 < 4) q0 = *(const uint4*)(q0src + ((jc + 1) << 11));
    __syncthreads();
    mma16(Hs, Ws, out);
  }
  // out: thread holds storage rows 64g + 4tx..+3
  const int r = 64 * g + tx * 4;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int m = ty * 4 + i, tile = tile0 + (m >> 4);
    if (tile >= P.T) continue;
    uint32_t o = 0;
#pragma unroll
    for (int e = 0; e < 4; e++) o |= to_e4m3(store8(out[i][e] * 256.f)) << (8 * e);
    *(uint32_t*)(P.out + (size_t)tile * 8192 + ch_off(r) + tok_off(m & 15)) = o;
  }
}

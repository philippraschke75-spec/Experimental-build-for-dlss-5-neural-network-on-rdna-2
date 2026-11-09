// Native drop-in for k_pre_block_1h_32_fp8 (block 0) on gfx1030. Same kernarg (PreParams, 0x50 B), grid
// W/8 x H/8 (224x128), but 64 threads per workgroup (one token per lane, see native-swin32).
// features (noise, colour, conditioning) -> lift 16->32 (f16, then e4m3) -> C=32 swin body (native-swin32)
// -> skip e4m3 at +0x08 in the 4x4-tile arena layout and the 2x2 pool e4m3 at +0x38 ([C/16][H/2][W/2][16]),
// both in the WMMA channel interleave PI. Feature/lift semantics were read off the emulator's LDS
// (emu_pre_window.py); layouts from native-prepost-post/POST_LAYOUT.md.
// Weights: the kernel decodes the DLL record at +0x10 itself (tables in pre_tables.h) into a module-global
// cache once per record pointer; the cache writes are inline asm so the compiler keeps scalar loads for it.
#include <hip/hip_runtime.h>
#include <stdint.h>
#ifndef PRE_ATTR
#define PRE_ATTR
#endif
#include "pre_tables.h"

typedef _Float16 h2 __attribute__((ext_vector_type(2)));

struct PreParams {
  const float* in;       // +0x00 f32 RGB, HWC, pitch W
  uint8_t* skip;         // +0x08
  const uint8_t* wrec;   // +0x10 block0.layer0 record
  int H, W;              // +0x18
  float cscale;          // +0x20 colour = (x - 0.5) * 2 * cscale
  int seed;              // +0x24
  float c28, c2c, c30;   // +0x28 +0x2c +0x30 conditioning features 11, 12, 10
  int pad34;             // +0x34 unused
  uint8_t* pool;         // +0x38
  const float* prev;     // +0x40 second colour source (null -> +0x00)
  float c48, c4c;        // +0x48 +0x4c conditioning features 13, 14
  const uint8_t* src;    // +0x50 game colour RGBA16F (src_w x src_h, pitch src_w * 8) or null (colour from +0x00): fused
  int src_h, src_w;      // +0x58 k_import (mode 0), src mirrored about its last row/column; post reads src the same way
};

#define W_EXP 0
#define W_CON (128 * 32)
#define W_QKV (W_CON + 32 * 128)
#define W_OUT (W_QKV + 96 * 32)
#define W_LIFT (W_OUT + 32 * 32)
#define W_HALFS (W_LIFT + 32 * 16)
#define F_BIAS 0
#define F_MS 4096
#define F_AS (4096 + 32)
#define F_HS (4096 + 64)
#define F_FLOATS (4096 + 65)

// __constant__ (addrspace 4) so the fast path's uniform loads are scalar; written only by decode_weights (asm stores)
__constant__ uint32_t gW[W_HALFS / 2];
__constant__ float gF[F_FLOATS];
__device__ unsigned long long gKey;

__device__ __forceinline__ void st32(void* p, uint32_t v) {   // invisible to alias analysis on purpose
  asm volatile("global_store_dword %0, %1, off" ::"v"(p), "v"(v));
}
__device__ __forceinline__ float e4m3(uint8_t b) {
  int e = (b >> 3) & 15, m = b & 7;
  float v = e ? ldexpf((float)(8 | m), e - 10) : ldexpf((float)m, -9);
  return (b & 0x80) ? -v : v;
}
__device__ __forceinline__ uint32_t hbits(_Float16 h) { return __builtin_bit_cast(uint16_t, h); }
__device__ __forceinline__ float rec_f16(const uint8_t* r, int off) {
  return (float)__builtin_bit_cast(_Float16, (uint16_t)(r[off] | (r[off + 1] << 8)));
}

// cache position -> record order. The contract weights are stored [hc/8][c][8] (record: [c][128]) so one MLP
// step reads 512 contiguous bytes (16 s_load_dwordx8) instead of 128 scattered s_load_dword.
__device__ __forceinline__ int wsrc(int e) {
  if (e < W_CON || e >= W_QKV) return e;
  int k = e - W_CON;
  return W_CON + ((k >> 3) & 31) * 128 + (k >> 8) * 8 + (k & 7);
}

__device__ void decode_weights(const uint8_t* r, int t) {
  for (int i = t; i < W_HALFS / 2; i += 64) {
    uint32_t lo, hi;
    if (2 * i < W_LIFT) { int e = wsrc(2 * i); lo = hbits((_Float16)e4m3(r[T_W[e]])); hi = hbits((_Float16)e4m3(r[T_W[e + 1]])); }
    else { lo = hbits((_Float16)rec_f16(r, T_LIFT[2 * i - W_LIFT])); hi = hbits((_Float16)rec_f16(r, T_LIFT[2 * i + 1 - W_LIFT])); }
    st32((void*)&gW[i], lo | (hi << 16));
  }
  for (int i = t; i < F_FLOATS; i += 64) {
    float v;
    if (i < F_MS) v = fmaf(rec_f16(r, T_BIAS[i]), 0.044921875f, 1.30078125f);
    else if (i < F_AS) v = rec_f16(r, T_MS[i - F_MS]);
    else if (i < F_HS) v = rec_f16(r, T_AS[i - F_AS]);
    else v = __builtin_bit_cast(float, (uint32_t)(r[T_HS] | (r[T_HS + 1] << 8) | (r[T_HS + 2] << 16) | ((uint32_t)r[T_HS + 3] << 24)));
    st32((void*)&gF[i], __float_as_uint(v));
  }
}

// ---- math helpers (as native-swin32)
__device__ __forceinline__ float q8(float x) {
#ifdef Q8FAST   // f16 + 3-bit mantissa only (2 int ops): keeps the look (CPU 45.3 dB to NVIDIA), no clamp (only swin needs it), no subnormal grid / ties-even
  { const uint32_t q_ = ((uint32_t)__builtin_bit_cast(unsigned short, (_Float16)x) + 0x40u) & 0xFF80u; return (float)__builtin_bit_cast(_Float16, (unsigned short)q_); }
#endif
  float a = fminf(fabsf(x), 448.f);
  uint32_t b = __float_as_uint(a);
  float rn = __uint_as_float((b + 0x7FFFFu + ((b >> 20) & 1u)) & 0xFFF00000u);
  float rs = __fsub_rn(__fadd_rn(a, 24576.f), 24576.f);
  return copysignf(a < 0.015625f ? rs : rn, x);
}
// byte code of an e4m3-grid value
__device__ __forceinline__ uint32_t e4m3_code(float v) {
#ifdef Q8BYTE   // f16 bits >> 7 - 64 (sat): tiny values below 2^-6 flush to 0
  { const uint32_t h_ = __builtin_bit_cast(unsigned short, (_Float16)v); const uint32_t a_ = min(h_ & 0x7FFFu, 0x5F00u) >> 7;
    return (a_ > 64u ? a_ - 64u : 0u) | ((h_ >> 8) & 0x80u); }
#endif
  uint32_t b = __float_as_uint(v), s = (b >> 24) & 0x80;
  float a = fabsf(v);
  if (a < 0.015625f) return s | (uint32_t)(a * 512.f);
  return s | ((((b >> 23) & 255) - 120) << 3) | ((b >> 20) & 7);
}
__device__ __forceinline__ float f16r(float x) { return (float)(_Float16)x; }
__device__ __forceinline__ float dot2(h2 a, h2 b, float c) { return __builtin_amdgcn_fdot2(a, b, c, false); }
__device__ __forceinline__ h2 pk(float a, float b) { h2 r; r.x = (_Float16)a; r.y = (_Float16)b; return r; }
// q8 of two f16 values at once (from post_native.hip; proof q8h_cpu.py): bit-equal to q8 on the f16 grid
__device__ __forceinline__ h2 q8h(h2 x) {
#ifdef Q8FAST
  return __builtin_bit_cast(h2, (__builtin_bit_cast(uint32_t, x) + 0x00400040u) & 0xFF80FF80u);
#endif
  uint32_t xb = __builtin_bit_cast(uint32_t, x);
  typedef unsigned short u2 __attribute__((ext_vector_type(2)));
  u2 ab = __builtin_elementwise_min(__builtin_bit_cast(u2, xb & 0x7FFF7FFFu), (u2){0x5F00, 0x5F00});
  u2 eb = __builtin_elementwise_max(ab & (u2){0x7C00, 0x7C00}, (u2){0x2400, 0x2400});
  h2 a = __builtin_bit_cast(h2, ab);
  h2 K = __builtin_bit_cast(h2, eb) * (h2){(_Float16)128.f, (_Float16)128.f};
  h2 r = (a + K) - K;
  return __builtin_bit_cast(h2, (__builtin_bit_cast(uint32_t, r) & 0x7FFF7FFFu) | (xb & 0x80008000u));
}
__device__ __forceinline__ h2 q8p(float a, float b) { return q8h(pk(a, b)); }   // q8(f16(a)), q8(f16(b)) packed

__device__ __forceinline__ float norm_scale(const float* v) {
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
  float r = (float)sqrt((double)(float)s);
  return __fdiv_rn(1.f, r);
}

// ---- noise: dlssnr.layers.pre_post.gaussian_noise (matches the kernel's LDS features)
__device__ __forceinline__ uint32_t pcg(uint32_t v) { v = (v >> ((v >> 28) + 4)) ^ v; return v * 0x108EF2D9u; }
__device__ __forceinline__ float unit(uint32_t s) { uint32_t t = pcg(s); return (float)(((t >> 30) ^ (t >> 8)) + 1u) * 5.9604644775390625e-8f; }

// k_import's padding: mirrored about the last row / column
__device__ __forceinline__ int mirror(int i, int n) { const int r = i < n ? i : 2 * n - i - 2; return r < 0 ? 0 : r > n - 1 ? n - 1 : r; }

// memory channel c holds reference channel PI(c)
__device__ __forceinline__ constexpr int PI(int c) { return (c & 1) | ((c >> 2) & 1) << 1 | ((c >> 3) & 1) << 2 | ((c >> 1) & 1) << 3 | (c & 16); }

template <class WP, class FP>
__device__ __forceinline__ void body(const PreParams& p, const int t, h2 (*Ks)[16], h2 (*Vt)[32], WP Wt, FP Fp) {
  const int W = p.W, H = p.H;
  const int py = blockIdx.y * 8 + ((t >> 5) & 1) * 4 + ((t >> 2) & 3);
  const int px = blockIdx.x * 8 + ((t >> 4) & 1) * 4 + (t & 3);

  // ---- 16 features, f16
  float f[16];
  {
    uint32_t base = ((uint32_t)px * 0x8DA6B343u) ^ ((uint32_t)p.seed * 0x9E3779B9u) ^ ((uint32_t)py * 0xD8163841u) ^ 0x243F6A88u;
    uint32_t tt = pcg(base), hh = (tt >> 22) ^ tt;
    float ua = unit(hh * 0x2C9277B5u + 0xAC564B05u), ub = unit(hh * 0xFA6DC5F9u + 0x4712A88Eu);
    float uc = unit(hh * 0xCAA5B80Du + 0x21DD796Bu), ud = unit(hh * 0x83232C31u + 0x3463E0ACu);
    float ra = sqrtf(-2.f * logf(ua)), rc = sqrtf(-2.f * logf(uc));
    float aa = ub * 6.28318530718f, ac = ud * 6.28318530718f;
    f[0] = f16r(ra * cosf(aa)); f[1] = f16r(ra * sinf(aa)); f[2] = f16r(rc * cosf(ac));
    f[3] = 1.f;
    size_t o = ((size_t)py * W + px) * 3;
    float c1[3];
    if (p.src) {   // fused k_import
      typedef _Float16 h4 __attribute__((ext_vector_type(4)));
      const h4 s = *(const h4*)(p.src + ((size_t)mirror(py, p.src_h) * p.src_w + mirror(px, p.src_w)) * 8);
      c1[0] = (float)s.x; c1[1] = (float)s.y; c1[2] = (float)s.z;
    } else {
#pragma unroll
      for (int i = 0; i < 3; i++) c1[i] = p.in[o + i];
    }
    float s2 = 2.f * p.cscale;
#pragma unroll
    for (int i = 0; i < 3; i++) {
      f[4 + i] = f16r(__fmul_rn(__fsub_rn(c1[i], 0.5f), s2));
      f[7 + i] = f16r(__fmul_rn(__fsub_rn(p.prev ? p.prev[o + i] : c1[i], 0.5f), s2));
    }
    f[10] = f16r(p.c30); f[11] = f16r(p.c28); f[12] = f16r(p.c2c); f[13] = f16r(p.c48); f[14] = f16r(p.c4c); f[15] = 0.f;
  }
  // ---- lift 16 -> 32 (kernel order: 4 fma, then 6 dot2), f16, then e4m3
  h2 xr[16];
  {
    h2 fp[6];
#pragma unroll
    for (int i = 0; i < 6; i++) fp[i] = pk(f[4 + 2 * i], f[5 + 2 * i]);
    float xl[32];
#pragma unroll
    for (int c = 0; c < 32; c++) {
      const _Float16* w = (const _Float16*)(Wt + (W_LIFT + c * 16) / 2);
      float s = fmaf(f[0], (float)w[0], 0.f);
      s = fmaf(f[1], (float)w[1], s); s = fmaf(f[2], (float)w[2], s); s = fmaf(f[3], (float)w[3], s);
#pragma unroll
      for (int i = 0; i < 6; i++) s = dot2(fp[i], Wt[(W_LIFT + c * 16 + 4) / 2 + i], s);
      xl[c] = s;
    }
#pragma unroll
    for (int i = 0; i < 16; i++) xr[i] = q8p(xl[2 * i], xl[2 * i + 1]);
  }
  const h2* xq = xr;   // already on the e4m3 grid

  // ---- MLP
  float y[32];
#pragma unroll
  for (int c = 0; c < 32; c++) y[c] = 0.f;
#pragma unroll 1
  for (int hc = 0; hc < 128; hc += 8) {
#ifdef FAST_ACT   // activation + e4m3 rounding of two hidden channels at once in packed f16 (not bit-exact: f16 intermediates)
    float sj[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      float s = 0.f;
#pragma unroll
      for (int k = 0; k < 16; k++) s = dot2(xq[k], Wt[(W_EXP + (hc + j) * 32) / 2 + k], s);
      sj[j] = s;
    }
    h2 ap[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const h2 s = pk(sj[2 * j], sj[2 * j + 1]);
      const h2 tc = __builtin_elementwise_min(__builtin_elementwise_max(s, (h2){(_Float16)-4.f, (_Float16)-4.f}), (h2){(_Float16)4.f, (_Float16)4.f});
      const h2 at = __builtin_bit_cast(h2, __builtin_bit_cast(uint32_t, tc) & 0x7FFF7FFFu);
      const h2 inner = at * (h2){(_Float16)-0.055908203125f, (_Float16)-0.055908203125f} + (h2){(_Float16)0.447265625f, (_Float16)0.447265625f};
      const h2 g = tc * inner + (h2){(_Float16)0.89453125f, (_Float16)0.89453125f};
      ap[j] = q8h(s * g);
    }
#else
    float a[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      float s = 0.f;
#pragma unroll
      for (int k = 0; k < 16; k++) s = dot2(xq[k], Wt[(W_EXP + (hc + j) * 32) / 2 + k], s);
      float tc = fminf(fmaxf(s, -4.f), 4.f);
      float inner = fmaf(fabsf(tc), -0.055908203125f, 0.447265625f);
      a[j] = q8(__fmul_rn(s, fmaf(tc, inner, 0.89453125f)));
    }
    h2 ap[4];
#pragma unroll
    for (int j = 0; j < 4; j++) ap[j] = pk(a[2 * j], a[2 * j + 1]);
#endif
#pragma unroll
    for (int c = 0; c < 32; c++)
#pragma unroll
      for (int j = 0; j < 4; j++) y[c] = dot2(ap[j], Wt[(W_CON + hc * 32 + c * 8) / 2 + j], y[c]);
  }
  h2 yq[16];
#pragma unroll
  for (int c = 0; c < 32; c += 2) {
    h2 xv = xr[c / 2];
    y[c] = __fadd_rn(__fmul_rn((float)xv.x, Fp[F_MS + c]), y[c]);
    y[c + 1] = __fadd_rn(__fmul_rn((float)xv.y, Fp[F_MS + c + 1]), y[c + 1]);
    yq[c / 2] = q8p(y[c], y[c + 1]);
  }

  // ---- qkv
  h2 qh[16];
  {
    float v[32];
#pragma unroll
    for (int pass = 0; pass < 3; pass++) {
#pragma unroll
      for (int d = 0; d < 32; d++) {
        float s = 0.f;
#pragma unroll
        for (int k = 0; k < 16; k++) s = dot2(yq[k], Wt[(W_QKV + (pass * 32 + d) * 32) / 2 + k], s);
        v[d] = s;
      }
      if (pass == 0) {
        float sc = __fmul_rn(norm_scale(v), Fp[F_HS]);
#pragma unroll
        for (int d = 0; d < 32; d += 2)
#ifdef FAST_ACT
          qh[d / 2] = q8p(__fmul_rn(v[d], sc), __fmul_rn(v[d + 1], sc));
#else
          qh[d / 2] = pk(q8(__fmul_rn(v[d], sc)), q8(__fmul_rn(v[d + 1], sc)));
#endif
      } else if (pass == 1) {
        float sc = norm_scale(v);
#pragma unroll
        for (int d = 0; d < 32; d += 2)
#ifdef FAST_ACT
          Ks[t][d / 2] = q8p(__fmul_rn(v[d], sc), __fmul_rn(v[d + 1], sc));
#else
          Ks[t][d / 2] = pk(q8(__fmul_rn(v[d], sc)), q8(__fmul_rn(v[d + 1], sc)));
#endif
      } else {
        _Float16* vt = (_Float16*)&Vt[0][0];
#pragma unroll
        for (int d = 0; d < 32; d += 2) { h2 r = q8p(v[d], v[d + 1]); vt[d * 64 + t] = r.x; vt[(d + 1) * 64 + t] = r.y; }
      }
    }
  }
  __syncthreads();

  // ---- softmax numerators, denominator, probabilities
  h2 pp[32];
  const float2* brow = (const float2*)(Fp + F_BIAS + t * 64);
#pragma unroll
  for (int j = 0; j < 32; j++) {
    float2 bb = brow[j];
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i++) { s0 = dot2(qh[i], Ks[2 * j][i], s0); s1 = dot2(qh[i], Ks[2 * j + 1][i], s1); }
    h2 r = pk(fmaf(s0, 0.044921875f, bb.x), fmaf(s1, 0.044921875f, bb.y));
    r = __builtin_elementwise_max(r, (h2){(_Float16)1.03125f, (_Float16)1.03125f});
    r = __builtin_elementwise_min(r, (h2){(_Float16)1.5693359375f, (_Float16)1.5693359375f});
    uint32_t bits = ((__builtin_bit_cast(uint32_t, r) & 0x03FF03FFu) << 5);
    pp[j] = __builtin_bit_cast(h2, bits);
  }
  h2 hm[2];
#pragma unroll
  for (int m = 0; m < 2; m++) {
    h2 pr[4];
#pragma unroll
    for (int q = 0; q < 4; q++) { int b = 2 * q + m; pr[q] = ((pp[b] + pp[8 + b]) + pp[16 + b]) + pp[24 + b]; }
    hm[m] = ((pr[0] + pr[1]) + pr[2]) + pr[3];
  }
  _Float16 den = (hm[0].x + hm[0].y) + (hm[1].x + hm[1].y);
  float rec = __fdiv_rn(1.f, fmaxf((float)den, 0.00006103515625f));
#pragma unroll
  for (int j = 0; j < 32; j++)
#ifdef FAST_ACT
    pp[j] = q8h(pp[j] * (h2){(_Float16)rec, (_Float16)rec});
#else
    pp[j] = q8p(__fmul_rn((float)pp[j].x, rec), __fmul_rn((float)pp[j].y, rec));
#endif

  h2 ch[16];
#pragma unroll
  for (int d = 0; d < 32; d += 2) {
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int j = 0; j < 32; j++) { s0 = dot2(pp[j], Vt[d][j], s0); s1 = dot2(pp[j], Vt[d + 1][j], s1); }
    ch[d / 2] = q8p(s0, s1);
  }

  float o[32];
#pragma unroll
  for (int c = 0; c < 32; c++) {
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < 16; k++) s = dot2(ch[k], Wt[(W_OUT + c * 32) / 2 + k], s);
    o[c] = __fadd_rn(__fmul_rn(y[c], Fp[F_AS + c]), s);
  }

  // finish the whole out projection before the stores: interleaved with them, its scalar weights stayed live
  // across the divergent store blocks and spilled (~380 SGPRs)
#pragma unroll
  for (int c = 0; c < 32; c++) asm volatile("" : "+v"(o[c]));
  // ---- skip: e4m3(f16(acc)), 4x4-pixel tiles of 512 B
  {
    const int pt = t & 15;
    uint8_t* dst = p.skip + ((size_t)(py >> 2) * (W >> 2) + (px >> 2)) * 512 + (pt & 7) * 64 + 4 * (pt >> 3);
#pragma unroll
    for (int hi = 0; hi < 2; hi++)
#pragma unroll
      for (int g = 0; g < 4; g++) {
        uint32_t w = 0;
#pragma unroll
        for (int b = 0; b < 4; b += 2) {
          h2 r = q8p(o[PI(16 * hi + 4 * g + b)], o[PI(16 * hi + 4 * g + b + 1)]);
          w |= (e4m3_code((float)r.x) | e4m3_code((float)r.y) << 8) << (8 * b);
        }
        *(uint32_t*)(dst + 16 * g + 8 * hi) = w;
      }
  }
  // ---- 2x2 pool of acc: column pairs in f32 -> f16, rows added in f16, quarter, e4m3
  {
    uint32_t w[8] = {};
#pragma unroll
    for (int m = 0; m < 32; m++) {
      const int c = PI(m);
      float pair = f16r(o[c] + __shfl_xor(o[c], 1));
      _Float16 ph = (_Float16)pair, other = (_Float16)__shfl_xor(pair, 4);
      _Float16 tot = (t & 4) ? (_Float16)(other + ph) : (_Float16)(ph + other);
      w[m >> 2] |= e4m3_code(q8((float)tot * 0.25f)) << (8 * (m & 3));
    }
    if ((t & 5) == 0) {
      const size_t plane = (size_t)(H >> 1) * (W >> 1) * 16;
      uint4* d0 = (uint4*)(p.pool + ((size_t)(py >> 1) * (W >> 1) + (px >> 1)) * 16);
      d0[0] = make_uint4(w[0], w[1], w[2], w[3]);
      *(uint4*)((uint8_t*)d0 + plane) = make_uint4(w[4], w[5], w[6], w[7]);
    }
  }
}

// marker for tools/build_plan.py: this k_pre_native reads the game colour itself (PreParams.src), no k_import step
extern "C" __global__ void k_pre_native_fused_io() {}
extern "C" __global__ __launch_bounds__(64) PRE_ATTR void k_pre_native(PreParams p) {
  __shared__ h2 Ks[64][16];
  __shared__ h2 Vt[32][32];
  const int t = threadIdx.x;
  // Fast path: the cache already holds this record. Plain (non-atomic) key load: it is no clobber for the
  // cache loads, so they stay scalar. The key is published only after the cache is complete.
  // First launch for this record: every such workgroup writes the same decoded values (idempotent), then
  // publishes the key. One body call for both paths (two inlined copies cost 608 SGPR spills).
  if (gKey != (unsigned long long)p.wrec) {
  decode_weights(p.wrec, t);
  asm volatile("s_waitcnt_vscnt null, 0x0\n\ts_waitcnt vmcnt(0)" ::: "memory");
  __syncthreads();
  if (t == 0) asm volatile("global_store_dwordx2 %0, %1, off\n\ts_waitcnt_vscnt null, 0x0" ::"v"(&gKey), "v"((unsigned long long)p.wrec) : "memory");
  asm volatile("buffer_gl0_inv\n\tbuffer_gl1_inv" ::: "memory");
  }
  // laundered pointers: loads through them cannot be hoisted above the decode
  typedef const __attribute__((address_space(4))) h2* CW;
  typedef const __attribute__((address_space(4))) float* CF;
  CW w; CF f;
  asm volatile("" : "=s"(w) : "0"((CW)gW));
  asm volatile("" : "=s"(f) : "0"((CF)gF));
  body(p, t, Ks, Vt, w, f);
}

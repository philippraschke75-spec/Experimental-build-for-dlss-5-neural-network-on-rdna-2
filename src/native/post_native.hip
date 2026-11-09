// Native fused k_post_block_1h_32_fp8 for gfx1030: blend (2x up of e4m3 decoder x + e4m3 skip, gains) ->
// C=32 Swin body (native-swin32) -> 32->4 projection -> compose to RGBA f32. Same PostParams kernarg as the
// original; launch 64 threads per 8x8 window, grid W/8 x H/8. Weights come from block70 at load (k_post_native_prep).
// Build: build.bat (-ffp-contract=off).
#include <hip/hip_runtime.h>
#include <stdint.h>
#ifndef POST_ATTR
#define POST_ATTR __attribute__((amdgpu_waves_per_eu(7)))   // perf-prepost: 144 VGPR, 7 waves/SIMD
#endif

typedef _Float16 h2 __attribute__((ext_vector_type(2)));

// weight blob (f16): expand[128][32], contract[128/8][32][8] (no BAKED: hc-chunk major), qkv[96][32], out[32][32]
#define W_EXP 0
#define W_CON (128 * 32)
#define W_QKV (W_CON + 32 * 128)
#define W_OUT (W_QKV + 96 * 32)
#define W_HALFS (W_OUT + 32 * 32)
// f32 blob: exp_bias[64][64], mlp_scale[32], attn_scale[32], head_scale
#define F_BIAS 0
#define F_MS 4096
#define F_AS (4096 + 32)
#define F_HS (4096 + 64)
#define F_FLOATS (4096 + 65)
#ifdef BAKED
#include "post_weights.h"   // old build: weights baked from block70 (vendor data, not committed) - only for A/B checks
#else
// Filled at load by k_post_native_prep from the block70.layer0 record (+0x18), so the .co ships no weights.
__constant__ uint16_t gWt[W_HALFS];
__constant__ float gFp[F_FLOATS];
__constant__ float gMain[32];
__constant__ float gSkip[32];
__constant__ uint16_t gProj[128];
#endif
#include "wstage.h"
#if PREW
#define WEXP(j, k) WEXP_L(Lr, j, k)
#define WCON(c, j) WCON_L(Lr, c, j)
#else
#define WEXP(j, k) Wt[WI((W_EXP + (hc + j) * 32) / 2 + k)]
#define WCON(c, j) Wt[WI((W_CON + hc * 32 + c * 8) / 2 + j)]   // contract cached [hc/8][c][8] by k_post_native_prep (perf-tile); PREW=1 staging assumes the old layout (off)
#endif

struct PostParams {
  const uint8_t* x; const uint8_t* skip; float* out; const uint8_t* wrec;   // wrec: block70.layer0 record (prep only)
  int H, W, col0, row0; float scale; int denorm; const float* frame; const float* aux; float gate;
  uint8_t* dst; int dst_h, dst_w;   // +0x50: fused k_export (mode 0) when dst != null: dst RGBA16F (pitch dst_w * 8) = f16(out)
  const uint8_t* src;               // +0x60: fused k_import when != null: frame = src (dst_w x dst_h RGBA16F) mirrored, as the pre-block
};

// nearest e4m3 (ties even), saturating at 448; branchless. Normal range: round the f32 mantissa to 3 bits;
// below 2^-6 the fixed 2^-9 step: adding 2^-9 * 1.5 * 2^23 (= 24576, ulp 2^-9) rounds to that step exactly.
__device__ __forceinline__ float q8(float x) {
#ifdef Q8FAST   // f16 + 3-bit mantissa only (2 int ops): keeps the look (CPU 45.3 dB to NVIDIA), no clamp (only swin needs it), no subnormal grid / ties-even
  { const uint32_t q_ = ((uint32_t)__builtin_bit_cast(unsigned short, (_Float16)x) + 0x40u) & 0xFF80u; return (float)__builtin_bit_cast(_Float16, (unsigned short)q_); }
#endif
#if KO_Q8
  return x;
#elif Q8F
  // same rounding, 7 VALU: magic-number add/sub with K = 2^(exp+20) (2^14 below 2^-6) rounds RNE to 2^(exp-3)
  float a = fminf(fabsf(x), 448.f);
  float e = __uint_as_float(__float_as_uint(a) & 0xFF800000u);
  float K = __fmul_rn(fmaxf(e, 0.015625f), 1048576.f);
  return copysignf(__fsub_rn(__fadd_rn(a, K), K), x);
#else
  float a = fminf(fabsf(x), 448.f);
  uint32_t b = __float_as_uint(a);
  float rn = __uint_as_float((b + 0x7FFFFu + ((b >> 20) & 1u)) & 0xFFF00000u);
  float rs = __fsub_rn(__fadd_rn(a, 24576.f), 24576.f);
  return copysignf(a < 0.015625f ? rs : rn, x);
#endif
}
__device__ __forceinline__ float f16r(float x) { return (float)(_Float16)x; }
__device__ __forceinline__ float dot2(h2 a, h2 b, float c) { return __builtin_amdgcn_fdot2(a, b, c, false); }
__device__ __forceinline__ h2 pk(float a, float b) { h2 r; r.x = (_Float16)a; r.y = (_Float16)b; return r; }
// q8 of two f16 values at once (packed f16 math): K = max(2^exp, 2^-6) * 2^7 gives f16 ulp 2^(exp-3) (2^-9 below 2^-6),
// so (a + K) - K is the same RNE as q8 (ties: both K have even mantissas). Only for inputs already on the f16 grid.
__device__ __forceinline__ h2 q8h(h2 x) {
#ifdef Q8FAST
  return __builtin_bit_cast(h2, (__builtin_bit_cast(uint32_t, x) + 0x00400040u) & 0xFF80FF80u);
#endif
#if (!defined(PKQ8) || PKQ8) && !KO_Q8
  uint32_t xb = __builtin_bit_cast(uint32_t, x);
  // |x| clamped at 448 and 2^exp floored at 2^-6 as unsigned 16-bit compares (same order as f16 for non-negative
  // values; NaN/inf sort above 448 like fminf), which avoids the canonicalising v_pk_max the f16 min/max emit
  typedef unsigned short u2 __attribute__((ext_vector_type(2)));
  u2 ab = __builtin_elementwise_min(__builtin_bit_cast(u2, xb & 0x7FFF7FFFu), (u2){0x5F00, 0x5F00});
  u2 eb = __builtin_elementwise_max(ab & (u2){0x7C00, 0x7C00}, (u2){0x2400, 0x2400});
  h2 a = __builtin_bit_cast(h2, ab);
  h2 K = __builtin_bit_cast(h2, eb) * (h2){(_Float16)128.f, (_Float16)128.f};
  h2 r = (a + K) - K;
  return __builtin_bit_cast(h2, (__builtin_bit_cast(uint32_t, r) & 0x7FFF7FFFu) | (xb & 0x80008000u));
#else
  return pk(q8((float)x.x), q8((float)x.y));
#endif
}
__device__ __forceinline__ h2 q8p(float a, float b) { return q8h(pk(a, b)); }   // q8(f16(a)), q8(f16(b)) packed
#if KO_W
#define WI(i) ((i) & 0x7FF)
#else
#define WI(i) (i)
#endif
#ifndef BAKED
// int8_prepost (opt-in, k_post_native_i8): the post-block MLP (expand 32->128, GELU, q8, contract 128->32) on v_dot4_i32_i8, as
// int8_swin does it: weights int8 with a power-of-two scale per output channel (built once by the module's _prep),
// activations int8 with a power-of-two scale per (token, 32-channel group). The inputs are on the e4m3 grid, so a
// value >= max/16 of its group is exact in int8; only the smaller ones round.
// Tables (filled by i8_tables in the _prep kernel, read as __constant__):
//   W: Ei[128 h][8] (word j = channels 4j..4j+3) | Ci[4 g][32 c][8] (word j = hidden 32g+4j..+3)
//   S: Es[128] | Cs[32] (row scales)
// Measured: post 1.42 -> 1.24 ms at 540p, +0.1 dB vs PyTorch. Not in the pre-block: there it costs 3.7 dB vs PyTorch
// (the error runs through the whole network) and was slower (0.94 -> 1.01 ms).
#define I8_CI (128 * 8)
#define I8_WORDS (I8_CI + 4 * 32 * 8)
#define I8_CS 128
#define I8_FLOATS (128 + 32)

// one row of K e4m3-grid weights -> int8 words at W[idx(j)] + power-of-two scale (max |w| < 2^e -> scale 2^(e-7))
template <int K, class F, class I>
__device__ float i8_row(F w, uint32_t* W, I idx) {
  float m = 0.f;
  for (int k = 0; k < K; k++) m = fmaxf(m, fabsf(w(k)));
  int e; frexpf(m, &e);
  const float s = m > 0.f ? ldexpf(1.f, e - 7) : 1.f;
  for (int j = 0; j < K / 4; j++) {
    uint32_t v = 0;
    for (int l = 0; l < 4; l++) v |= ((uint32_t)(int)rintf(w(4 * j + l) / s) & 0xFFu) << (8 * l);
    W[idx(j)] = v;
  }
  return s;
}
// expand weight (h, k) = we(h, k), contract weight (c, h) = wc(c, h); thread t of nt
template <class FE, class FC>
__device__ void i8_tables(FE we, FC wc, uint32_t* W, float* S, int t, int nt) {
  for (int r = t; r < 160; r += nt) {
    if (r < 128) S[r] = i8_row<32>([&](int k) { return we(r, k); }, W, [&](int j) { return r * 8 + j; });
    else { const int c = r - 128;
      S[I8_CS + c] = i8_row<128>([&](int h) { return wc(c, h); }, W, [&](int j) { return I8_CI + (j >> 3) * 256 + c * 8 + (j & 7); }); }
  }
}

// int8 of 4 values already scaled to |v| < 128: the low byte of v + 1.5*2^23 is rint(v) in two's complement
__device__ __forceinline__ uint32_t i8x4(float a, float b, float c, float d, float inv) {
  const float M = 12582912.f;
  uint32_t A = __float_as_uint(fmaf(a, inv, M)), B = __float_as_uint(fmaf(b, inv, M));
  uint32_t C = __float_as_uint(fmaf(c, inv, M)), D = __float_as_uint(fmaf(d, inv, M));
  return __builtin_amdgcn_perm(__builtin_amdgcn_perm(D, C, 0x0c0c0400u), __builtin_amdgcn_perm(B, A, 0x0c0c0400u), 0x05040100u);
}
// power-of-two scale for a group with max |v| = m: inv = 2^(6-E), scale = 2^(E-6) for m in [2^E, 2^(E+1)); m == 0 -> anything
__device__ __forceinline__ void i8_scale(float m, float& inv, float& sc) {
  const uint32_t eb = __float_as_uint(fmaxf(m, 1e-30f)) >> 23;
  inv = __uint_as_float((260u - eb) << 23); sc = __uint_as_float((eb - 6u) << 23);
}

// y[c] = sum_h q8(gelu(x . We[h])) * Wc[c][h]  (same GELU / q8 as the f16 path); xq: 32 e4m3-grid channels
template <class WP, class SP>
__device__ __forceinline__ void mlp_i8(const h2* xq, float* y, WP W, SP S) {
  uint32_t xi[8]; float sa;
  {
    float m = 0.f;
#pragma unroll
    for (int k = 0; k < 16; k++) m = fmaxf(m, fmaxf(fabsf((float)xq[k].x), fabsf((float)xq[k].y)));
    float inv; i8_scale(m, inv, sa);
#pragma unroll
    for (int j = 0; j < 8; j++) xi[j] = i8x4(xq[2 * j].x, xq[2 * j].y, xq[2 * j + 1].x, xq[2 * j + 1].y, inv);
  }
#pragma unroll
  for (int c = 0; c < 32; c++) y[c] = 0.f;
#pragma unroll 1
  for (int g = 0; g < 4; g++) {
    float a[32], m = 0.f;
#pragma unroll
    for (int j = 0; j < 32; j++) {
      const int h = 32 * g + j;
      int acc = 0;
#pragma unroll
      for (int k = 0; k < 8; k++) acc = __builtin_amdgcn_sdot4((int)xi[k], (int)W[h * 8 + k], acc, false);
      float s = (float)acc * (sa * S[h]);
      float tc = fminf(fmaxf(s, -4.f), 4.f);
      float inner = fmaf(fabsf(tc), -0.055908203125f, 0.447265625f);
      a[j] = q8(__fmul_rn(s, fmaf(tc, inner, 0.89453125f)));
      m = fmaxf(m, fabsf(a[j]));
    }
    float inv, sb; i8_scale(m, inv, sb);
    uint32_t ai[8];
#pragma unroll
    for (int j = 0; j < 8; j++) ai[j] = i8x4(a[4 * j], a[4 * j + 1], a[4 * j + 2], a[4 * j + 3], inv);
#pragma unroll
    for (int c = 0; c < 32; c++) {
      int acc = 0;
#pragma unroll
      for (int j = 0; j < 8; j++) acc = __builtin_amdgcn_sdot4((int)ai[j], (int)W[I8_CI + g * 256 + c * 8 + j], acc, false);
      y[c] = fmaf((float)acc, sb, y[c]);
    }
  }
#pragma unroll
  for (int c = 0; c < 32; c++) y[c] *= S[I8_CS + c];
}
__constant__ uint32_t gI8W[I8_WORDS];
__constant__ float gI8S[I8_FLOATS];
#endif

// reciprocal L2 norm with the reference's chain/half summation order (d = 16 frag + 8 group + 2 chain + half)
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

// e4m3 byte -> float (exact): place in f16 bits and scale by 2^8
__device__ __forceinline__ float e4(uint32_t b) {
  uint16_t h = (uint16_t)(((b & 0x80u) << 8) | ((b & 0x7Fu) << 7));
  return (float)__builtin_bit_cast(_Float16, h) * 256.f;
}
// memory channel c holds reference channel PI(c)
__device__ constexpr int PI(int c) { return (c & 1) | ((c >> 2) & 1) << 1 | ((c >> 3) & 1) << 2 | ((c >> 1) & 1) << 3 | (c & 16); }

// compose kept out of line: inlined, the uniform branches made the scheduler spill the whole body. Fields by value:
// with a PostParams reference the kernel copied the struct to scratch and the call read it back (NR_PROF post 1.45 -> 1.02 ms at 540p).
__device__ __noinline__ void compose(float* out, uint8_t* dst, const uint8_t* src, const float* frame, const float* aux, float scale,
                                     int denorm, float gate, int W, int dh, int dw, int py, int px, float r0, float r1, float r2, float r3) {
  const size_t pix = (size_t)py * W + px;
  const float r[4] = {r0, r1, r2, r3};
  float f[3] = {0.f, 0.f, 0.f};
  if (src) {
    auto mirror = [](int i, int n) { const int r = i < n ? i : 2 * n - i - 2; return r < 0 ? 0 : r > n - 1 ? n - 1 : r; };
    typedef _Float16 h4 __attribute__((ext_vector_type(4)));
    const h4 s = *(const h4*)(src + ((size_t)mirror(py, dh) * dw + mirror(px, dw)) * 8);
    f[0] = (float)s.x; f[1] = (float)s.y; f[2] = (float)s.z;
  } else if (frame) { f[0] = frame[pix * 3]; f[1] = frame[pix * 3 + 1]; f[2] = frame[pix * 3 + 2]; }
  float res[3];
#pragma unroll
  for (int k = 0; k < 3; k++) res[k] = 0.125f * f[k] - 0.0625f + scale * r[k];
  float alpha = 0.f;
  if (denorm) {
    alpha = 1.f;
#pragma unroll
    for (int k = 0; k < 3; k++) res[k] = fminf(fmaxf(8.f * res[k] + 0.5f, 0.f), 1.f);
    if (aux && gate > 0.f) {
      float g = fminf(fmaxf(gate / (1.f + __expf(-r[3])), 0.f), 1.f);
#pragma unroll
      for (int k = 0; k < 3; k++) res[k] = res[k] + g * (aux[pix * 3 + k] - res[k]);
    }
  }
  ((float4*)out)[pix] = make_float4(res[0], res[1], res[2], alpha);
  if (dst && py < dh && px < dw) {
    typedef _Float16 h4 __attribute__((ext_vector_type(4)));
    *(h4*)(dst + ((size_t)py * dw + px) * 8) = (h4){(_Float16)res[0], (_Float16)res[1], (_Float16)res[2], (_Float16)alpha};
  }
}

template <bool I8>
__device__ __forceinline__ void post_main(const PostParams& P) {
  __shared__ h2 Ks[64][16];
  __shared__ h2 Vt[32][32];
  const h2* Wt = (const h2*)gWt; const float* Fp = gFp;
  const int t = threadIdx.x, H = P.H, W = P.W;
  const int py = P.row0 + blockIdx.y * 8 + ((t >> 5) & 1) * 4 + ((t >> 2) & 3);
  const int px = P.col0 + blockIdx.x * 8 + ((t >> 4) & 1) * 4 + (t & 3);
  const bool inside = py >= 0 && py < H && px >= 0 && px < W;

  // ---- blend: b = f16(skip*skip_gain + f16(up(x)*main_gain)), memory channel order -> ref slots
  h2 xr[16], xq[16];
  {
    float xb[32];
#pragma unroll
    for (int c = 0; c < 32; c++) xb[c] = 0.f;
    if (inside) {
      const int hw = (H >> 1) * (W >> 1);
      const uint4* xs = (const uint4*)(P.x + ((size_t)(py >> 1) * (W >> 1) + (px >> 1)) * 16);
      uint4 u0 = xs[0], u1 = *(const uint4*)((const uint8_t*)xs + (size_t)hw * 16);
      uint32_t xw[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
      const int p = (py & 3) * 4 + (px & 3);
      const uint8_t* sb = P.skip + ((size_t)(py >> 2) * (W >> 2) + (px >> 2)) * 512 + (p & 7) * 64 + 4 * (p >> 3);
      uint32_t sw[8];
#pragma unroll
      for (int k = 0; k < 4; k++) { sw[k] = *(const uint32_t*)(sb + 16 * k); sw[4 + k] = *(const uint32_t*)(sb + 16 * k + 8); }
#pragma unroll
      for (int c = 0; c < 32; c++) {
        float up = e4((xw[c >> 2] >> (8 * (c & 3))) & 0xFF);
        float sk = e4((sw[c >> 2] >> (8 * (c & 3))) & 0xFF);
        float m = f16r(__fmul_rn(up, gMain[c]));
        xb[PI(c)] = f16r(fmaf(sk, gSkip[c], m));
      }
    }
#pragma unroll
    for (int i = 0; i < 16; i++) xr[i] = pk(xb[2 * i], xb[2 * i + 1]);
  }
#pragma unroll
  for (int i = 0; i < 16; i++) xq[i] = q8h(xr[i]);

  // ---- MLP: expand 32->128 (+cubic SiLU, q8), contract 128->32, streamed in chunks of 8 hidden values
  float y[32];
#ifndef BAKED
  if constexpr (I8) mlp_i8(xq, y, (const uint32_t*)gI8W, (const float*)gI8S);
  else
#endif
  {
#pragma unroll
  for (int c = 0; c < 32; c++) y[c] = 0.f;
#if PREW
  __shared__ uint4 Wl[2][64];
  Wl[0][t] = wfetch(Wt, 0, t);
  PREW_SYNC();
#endif
#pragma unroll 1
  for (int hc = 0; hc < 128; hc += 8) {
#if PREW
    const LP Lc = (LP)Wl[(hc >> 3) & 1];
    LP Lr;
    float pin = 0.f;
    uint4 wn;
    if (hc + 8 < 128) wn = wfetch(Wt, hc + 8, t);   // next chunk in flight during this one
#endif
#if defined(FAST_ACT) && !defined(NO_FAST_MLP) && !PREW   // activation + e4m3 rounding of two hidden channels at once in packed f16 (not bit-exact)
    float sj[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      float s = 0.f;
#pragma unroll
      for (int k = 0; k < 16; k++) s = dot2(xq[k], WEXP(j, k), s);
      sj[j] = s;
    }
    h2 ap[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const h2 s = pk(sj[2*j], sj[2*j+1]);
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
      PREW_ROW();
      float s = 0.f;
#pragma unroll
      for (int k = 0; k < 16; k++) s = dot2(xq[k], WEXP(j, k), s);
      float tc = fminf(fmaxf(s, -4.f), 4.f);
      float inner = fmaf(fabsf(tc), -0.055908203125f, 0.447265625f);
      a[j] = q8(__fmul_rn(s, fmaf(tc, inner, 0.89453125f)));
      PREW_DONE(a[j]);
    }
    h2 ap[4];
#pragma unroll
    for (int j = 0; j < 4; j++) ap[j] = pk(a[2*j], a[2*j+1]);
#endif
#pragma unroll
    for (int c = 0; c < 32; c++) {
      PREW_ROW();
#pragma unroll
      for (int j = 0; j < 4; j++) y[c] = dot2(ap[j], WCON(c, j), y[c]);
      PREW_DONE(y[c]);
    }
#if PREW
    if (hc + 8 < 128) Wl[((hc >> 3) & 1) ^ 1][t] = wn;
    PREW_SYNC();   // also: everyone is done reading this buffer before it is refilled next iteration
#endif
  }
  }
  // y = rs * x + mlp (two roundings), skip = y in f32, matrix operand = q8(f16(y))
  h2 yq[16];
#pragma unroll
  for (int c = 0; c < 32; c += 2) {
    h2 xv = xr[c / 2];
    y[c] = __fadd_rn(__fmul_rn((float)xv.x, Fp[F_MS + c]), y[c]);
    y[c+1] = __fadd_rn(__fmul_rn((float)xv.y, Fp[F_MS + c + 1]), y[c+1]);
    yq[c / 2] = q8p(y[c], y[c+1]);
  }

  // ---- qkv in three passes of 32
  h2 qh[16];
  {
    float v[32];
#pragma unroll
    for (int p = 0; p < 3; p++) {
#pragma unroll
      for (int d = 0; d < 32; d++) {
        float s = 0.f;
#pragma unroll
        for (int k = 0; k < 16; k++) s = dot2(yq[k], Wt[WI((W_QKV + (p * 32 + d) * 32) / 2 + k)], s);
        v[d] = s;
      }
      if (p == 0) {
        float sc = __fmul_rn(norm_scale(v), Fp[F_HS]);
#pragma unroll
        for (int d = 0; d < 32; d += 2)
#if defined(FAST_ACT) && !defined(NO_FAST_ATT)
          qh[d / 2] = q8p(__fmul_rn(v[d], sc), __fmul_rn(v[d+1], sc));
#else
          qh[d / 2] = pk(q8(__fmul_rn(v[d], sc)), q8(__fmul_rn(v[d+1], sc)));
#endif
      } else if (p == 1) {
        float sc = norm_scale(v);
#pragma unroll
        for (int d = 0; d < 32; d += 2)
#if defined(FAST_ACT) && !defined(NO_FAST_ATT)
          Ks[t][d / 2] = q8p(__fmul_rn(v[d], sc), __fmul_rn(v[d+1], sc));
#else
          Ks[t][d / 2] = pk(q8(__fmul_rn(v[d], sc)), q8(__fmul_rn(v[d+1], sc)));
#endif
      } else {
        _Float16* vt = (_Float16*)&Vt[0][0];
#pragma unroll
        for (int d = 0; d < 32; d += 2) { h2 vq = q8p(v[d], v[d+1]); vt[d * 64 + t] = vq.x; vt[(d+1) * 64 + t] = vq.y; }
      }
    }
  }
  __syncthreads();

  // ---- logits -> exponent-ramp numerators (f16)
  h2 pp[32];   // numerators, then probabilities; key pairs (2j, 2j+1)
  const float2* brow = (const float2*)(Fp + F_BIAS + t * 64);
#pragma unroll
  for (int j = 0; j < 32; j++) {
    float2 bb = brow[j];
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i++) { s0 = dot2(qh[i], Ks[2*j][i], s0); s1 = dot2(qh[i], Ks[2*j+1][i], s1); }
    h2 r = pk(fmaf(s0, 0.044921875f, bb.x), fmaf(s1, 0.044921875f, bb.y));
    r = __builtin_elementwise_max(r, (h2){(_Float16)1.03125f, (_Float16)1.03125f});
    r = __builtin_elementwise_min(r, (h2){(_Float16)1.5693359375f, (_Float16)1.5693359375f});
    uint32_t bits = ((__builtin_bit_cast(uint32_t, r) & 0x03FF03FFu) << 5);
    pp[j] = __builtin_bit_cast(h2, bits);
  }
  // denominator: key = 16 block + 4 pair + 2 member + half; blocks, then pairs, then halves, then members.
  // pp[j] holds keys 2j (.x, half 0) and 2j+1 (.y, half 1); key/2 = 8 block + 2 pair + member.
  h2 hm[2];
#pragma unroll
  for (int m = 0; m < 2; m++) {
    h2 pr[4];
#pragma unroll
    for (int p = 0; p < 4; p++) { int b = 2 * p + m; pr[p] = ((pp[b] + pp[8 + b]) + pp[16 + b]) + pp[24 + b]; }
    hm[m] = ((pr[0] + pr[1]) + pr[2]) + pr[3];
  }
  _Float16 den = (hm[0].x + hm[0].y) + (hm[1].x + hm[1].y);
  float rec = __fdiv_rn(1.f, fmaxf((float)den, 0.00006103515625f));
#pragma unroll
  for (int j = 0; j < 32; j++)
#if defined(FAST_ACT) && !defined(NO_FAST_ATT)
    pp[j] = q8h(pp[j] * (h2){(_Float16)rec, (_Float16)rec});
#else
    pp[j] = q8p(__fmul_rn((float)pp[j].x, rec), __fmul_rn((float)pp[j].y, rec));
#endif

  // ---- context = P V, stored q8(f16())
  h2 ch[16];
#pragma unroll
  for (int d = 0; d < 32; d += 2) {
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int j = 0; j < 32; j++) { s0 = dot2(pp[j], Vt[d][j], s0); s1 = dot2(pp[j], Vt[d+1][j], s1); }
    ch[d / 2] = q8p(s0, s1);
  }

  // ---- out = ars * skip + Wo . context ; written as the f32 accumulator
  if (!inside) return;
  // out = ars * skip + Wo . context; wide = f16(out) streamed straight into the 32->4 projection (f32 accumulate)
  const h2* pr = (const h2*)gProj;
  float r[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int c = 0; c < 32; c += 2) {
    float o2[2];
#pragma unroll
    for (int e = 0; e < 2; e++) {
      float s = 0.f;
#pragma unroll
      for (int k = 0; k < 16; k++) s = dot2(ch[k], Wt[WI((W_OUT + (c + e) * 32) / 2 + k)], s);
      o2[e] = __fadd_rn(__fmul_rn(y[c + e], Fp[F_AS + c + e]), s);
    }
    h2 wv = pk(o2[0], o2[1]);
#pragma unroll
    for (int k = 0; k < 4; k++) r[k] = dot2(wv, pr[k * 16 + c / 2], r[k]);
  }
  compose(P.out, P.dst, P.src, P.frame, P.aux, P.scale, P.denorm, P.gate, W, P.dst_h, P.dst_w, py, px, r[0], r[1], r[2], r[3]);
}
// marker for tools/build_plan.py: this k_post_native writes the game output itself (PostParams.dst), no k_export step
extern "C" __global__ void k_post_native_fused_io() {}
extern "C" __global__ __launch_bounds__(64) POST_ATTR void k_post_native(PostParams P) { post_main<false>(P); }
#ifndef BAKED
// int8_prepost: MLP on int8 dot4 (mlp_i8 above), tables from k_post_native_prep
extern "C" __global__ __launch_bounds__(64) POST_ATTR void k_post_native_i8(PostParams P) { post_main<true>(P); }
#endif

#ifndef BAKED
// ---- load-time prep: decode the block70.layer0 record into the __constant__ tables above. The runtime launches it
// once per module with the step's kernarg, grid 1 x 256 threads. Same values as the old baked post_weights.h
// (dlssnr-pytorch gen_weights.py); record layout from dlssnr-pytorch extract.py (_post_block_layout).
#define R_EXP 0x0000    // e4m3 [128][32], packed, inner 8
#define R_CON 0x1000    // e4m3 [32][128], inner 2
#define R_RS 0x2010     // residual_scale f16[32] (mlp scale; not zero-padded in this record)
#define R_MAIN 0x2050   // main_gain f16[32]
#define R_SKIP 0x2090   // skip_gain f16[32]
#define R_QKV 0x20d0    // e4m3 [96][32], inner 6
#define R_BIAS 0x2cd0   // attn_pos_bias f16[64*64], fragment order
#define R_SB 0x4cd0     // scalars_b f32 (head scale)
#define R_OUT 0x4ce0    // e4m3 [32][32], inner 2
#define R_ARS 0x50e0    // attn_residual_scale f16[32]
#define R_PROJ 0x5130   // out_project f16, two 16-input operand blocks

// byte of element (r, c) of a row-major [rows][cols] matrix in its packed 512-byte slices (extract.py unswizzle_matrix)
__device__ int unswz(int r, int c, int cols, int inner) {
  const int n = r & 7, half = (r >> 3) & 1, t = (r >> 4) % inner, o = (r >> 4) / inner;
  const int p = c & 1, j = (c >> 1) & 3, q = (c >> 3) & 3, g = c >> 5;
  return ((((((o * (cols >> 5) + g) * inner + t) * 8 + n) * 4 + j) * 2 + half) * 4 + q) * 2 + p;
}
// writable alias of a __constant__ table: the asm hides where the address came from, else the stores are dropped
template <class T> __device__ T* wr(T* p) { uint64_t a = (uint64_t)p; asm volatile("" : "+s"(a)); return (T*)a; }
__device__ float rf16(const uint8_t* R, int off, int i) { return (float)__builtin_bit_cast(_Float16, *(const uint16_t*)(R + off + 2 * i)); }

extern "C" __global__ __launch_bounds__(256) void k_post_native_prep(PostParams P) {
  const uint8_t* R = P.wrec; const int t = threadIdx.x;
  uint16_t* const gWt = wr(::gWt); uint16_t* const gProj = wr(::gProj);
  float* const gFp = wr(::gFp); float* const gMain = wr(::gMain); float* const gSkip = wr(::gSkip);
  for (int i = t; i < W_HALFS; i += 256) {
    int off, k, cols, inner;
    if (i < W_CON) { off = R_EXP; k = i; cols = 32; inner = 8; }
    else if (i < W_QKV) {   // contract cached [hc/8][c][8]: one MLP step reads 512 contiguous bytes
      off = R_CON; k = i - W_CON; cols = 128; inner = 2; k = ((k >> 3) & 31) * 128 + (k >> 8) * 8 + (k & 7);
    }
    else if (i < W_OUT) { off = R_QKV; k = i - W_QKV; cols = 32; inner = 6; }
    else { off = R_OUT; k = i - W_OUT; cols = 32; inner = 2; }
    gWt[i] = __builtin_bit_cast(uint16_t, (_Float16)e4(R[off + unswz(k / cols, k % cols, cols, inner)]));
  }
  for (int k = t; k < 64 * 64; k += 256) {   // exp_bias = fma(bias, slope, offset) in f64, rounded once to f32
    const int i = k >> 6, j = k & 63;
    const int lane = 4 * (i & 7) + (j & 7) / 2, slot = (j & 1) | (((i & 15) >> 3) << 1) | (((j & 15) >> 3) << 2);
    gFp[F_BIAS + k] = (float)((double)rf16(R, R_BIAS, (4 * (i >> 4) + (j >> 4)) * 256 + lane * 8 + slot) * 0.044921875 + 1.30078125);
  }
  if (t < 32) {
    gFp[F_MS + t] = rf16(R, R_RS, t); gFp[F_AS + t] = rf16(R, R_ARS, t);
    gMain[t] = rf16(R, R_MAIN, PI(t)); gSkip[t] = rf16(R, R_SKIP, PI(t));
  }
  if (t == 0) gFp[F_HS] = *(const float*)(R + R_SB);
  if (t < 128) {   // rows 0..3 (RGB + gate) of the 16x32 projection
    const int n = t >> 5, k = t & 31;
    gProj[t] = ((const uint16_t*)(R + R_PROJ))[(k >> 4) * 256 + 32 * (n & 7) + 8 * ((k & 7) >> 1) + 4 * (n >> 3) + (k & 1) + 2 * ((k & 15) >> 3)];
  }
  i8_tables([&](int h, int k) { return e4(R[R_EXP + unswz(h, k, 32, 8)]); }, [&](int c, int h) { return e4(R[R_CON + unswz(c, h, 128, 2)]); },
            wr(gI8W), wr(gI8S), t, 256);
}
#endif

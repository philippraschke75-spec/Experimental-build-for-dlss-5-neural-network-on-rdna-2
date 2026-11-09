// Native gfx1030 multi-head Swin block (Swin2H/4H/8H, swin.py) on the arena layout of k_swin_var<C>, C = 64/128/256.
// Drop-ins for _Z10k_swin_varILi{64,128,256}ELb0EEv9VarParams: same VarParams kernarg, same grid (one 8x8 window per
// workgroup), 256 threads. A window's whole block stays in LDS; weights are read straight from the e4m3 record and
// converted to f16 (x 2^-8, undone in the epilogues) while being staged into LDS.
// Rounding points follow dlssnr/layers/swin.py. Build with -ffp-contract=off. Layouts: native-mid/RESULTS.md.
//  activation (inside a stage): e4m3, 4x4-pixel tiles of C*16 B, tile = (row/4)*(W/4) + col/4, C/32 chunks of 512 B,
//     byte = chunk*512 + (p%8)*64 + 4*(p>=8) + 16*((c%16)/4) + 8*(c>=16) + (c&3), p = (row%4)*4 + col%4,
//     reference channel = 32*chunk + PI[c], PI = c0|c2<<1|c3<<2|c1<<3|c4<<4.
//  between stages (flags 1 in, 2 out, 4 -> +0x38, 8 <- +0x30): planes [C/16][H][W][16], same interleave.
//  weight matrix [N][K]: 16x32 tiles, tile = g*(GN*K/32) + kb*GN + j (n-tile = g*GN + j),
//     byte in tile = (n&7)*64 + ((k>>1)&3)*16 + (n>>3)*8 + (k>>3)*2 + (k&1).
#include <hip/hip_runtime.h>
#include <stdint.h>
#include "nr_params.h"   // kernarg structs: one definition, shared with tools/build_plan.py

#ifndef KPU   // k-loop unroll: 4 (C >= 128), 2 (C = 64, fewer live registers)
#define KPU (C == 64 ? 2 : 4)
#endif
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
__device__ __forceinline__ float pow2s(float m) {   // power-of-two int8 scale: m * 2^(7-e) < 128
  int e; frexpf(m, &e);
  return m > 0.f ? ldexpf(1.f, e - 7) : 1.f;
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

// SPLIT (perf-swin2, launched with grid z = 2): workgroup z computes expand+mid (Ms) of heads [z*NH/2, (z+1)*NH/2) and
// stores Ms in g_ms; the workgroup of the window that arrives second runs the contract over all NH heads in order
// 0..NH-1 from g_ms, then the rest of the block. Same operations in the same order as SPLIT = false (byte-identical).
// ponytail: fixed window capacity per width; windows beyond it run unsplit in workgroup z = 0.
constexpr int CAP256 = 160, CAP128 = 512;
__device__ uint32_t g_ms256[CAP256 * 8 * 1024], g_ms128[CAP128 * 4 * 1024];
__device__ uint32_t g_cnt256[CAP256], g_cnt128[CAP128];

// M (int8_swin mask): GEMMs that run on v_dot4_i32_i8 instead of f16 dot2. Weights: int8 with a power-of-two scale per
// output channel, built once at load by k_*_i8_prep into P.scratch (staged LDS order, then float scales). Activations:
// int8 with a power-of-two scale per (token, 32-channel group), made from the e4m3-grid values right where they are produced.
//  bit 0 (1): MLP expand + qkv (Xq/Yq made once per window; x itself is kept as e4m3 bytes, exact, for the residual)
//  bit 1 (2): MLP mid (Hs quantised in the expand epilogue)   bit 2 (4): MLP contract (Ms quantised in the mid epilogue)
//  bit 3 (8): attention out_proj (ctx quantised after P V)
// Scratch (bytes): expand int8 4C^2 | f32[4C] | qkv int8 3C^2 | f32[3C] | mid int8 128C | f32[C] | contract C^2 | f32[C] |
// out_proj C^2 | f32[C] (offsets I8Off).
template <int C> struct I8Off {
  static constexpr int Q = 4 * C * C + 16 * C, M = Q + 3 * C * C + 12 * C, K = M + 132 * C, O = K + C * C + 4 * C, END = O + C * C + 4 * C;
};
template <int C, bool SPLIT = false, int M = 0>
__device__ __forceinline__ void swin_block(const VarParams& P) {
  constexpr bool I8 = M & 1, MID = M & 2, CON = M & 4, OUTP = M & 8;
  static_assert(!(M & ~15) && (M == 0 || I8), "int8 mask: bit 0 always with the others");
  static_assert(I8Off<C>::END <= (1 << 20), "int8 scratch is 1 MiB per record (nr_runtime)");
  using OF = I8Off<C>;
  constexpr int NH = C / 32, KB = C / 32;                  // heads, 32-wide K steps over C
  constexpr int PWD = C >= 128 ? 128 : 64, NPASS = C / PWD, CPT = PWD / 32;   // C-wide outputs: tx*CPT within a pass
  // SMALL (C = 64): 28 KB of LDS so that two workgroups fit a CU. The hidden chunk is 64 instead of 128, q/k/Pm/ctx
  // are spread over R1 and R3 (see the attention phase), and the softmax reciprocals live in R1.
  constexpr bool SMALL = C == 64;
  constexpr int HC = SMALL ? 64 : 128, NS = 128 / HC, HCPT = HC / 32;   // hidden chunk, sub-chunks per head
  __shared__ uint32_t Xs[C / 2 * 64];          // [kp][tok]: x, later Yq (attention input and skip)
  __shared__ uint32_t R1[SMALL ? 2048 : 4096];  // Hs[HC/2 kp][64] | q (+k) f32 | Pm | ctx/Wo | coarse/pooled
  __shared__ uint32_t R2[1024];                // Ms[16 kp][64 tok] | Vt f16 [32 d][64 key]
  __shared__ uint32_t R3[2048];                // weight stage | qn,kn or k | Pm | ctx | Pj f16 [16][C]
  __shared__ float rec_arr[SMALL ? 1 : 64];
  __shared__ float sa_s[I8 ? 64 * KB : 1];   // I8: activation scale per (32-channel group, token)
  float* rec_s = SMALL ? (float*)(R1 + 1024) : rec_arr;

  const int tid = threadIdx.x;
  const uint8_t* rec = P.w;
  const int H = P.H, W = P.W, WT = W >> 2;
  const int r0 = P.Y0 + blockIdx.y * 8, c0 = P.X0 + blockIdx.x * 8;
  const bool up = P.flags & 8;
  // record offsets (dlssnr extract.py _swin_layout)
  constexpr int R_EXP = 0, R_MID = 4 * C * C, R_CON = R_MID + C * 128, R_AFT = R_CON + C * C;
  const int R_RS = up ? R_AFT + 2 * C * C : R_AFT;
  const int R_QKV = R_RS + 2 * C + 32 + (up ? 2 * C - 32 : 0);
  const int R_BIAS = R_QKV + 3 * C * C, R_HS = R_BIAS + NH * 8192, R_OUT = R_HS + (4 * NH > 16 ? 4 * NH : 16);
  const int R_ARS = R_OUT + C * C, R_DSRS = R_ARS + 2 * C;
  const _Float16* rs = (const _Float16*)(rec + R_RS);
  const uint4 rs8 = *(const uint4*)rs;
  const int rso = (rs8.x | rs8.y | rs8.z | rs8.w) == 0 ? 8 : 0;   // zero-padded table: MLP scales start at 8
  _Float16* Pj = (_Float16*)R3;
  const int tx = tid & 31, ty = tid >> 5;   // C-wide GEMM tiles: tokens ty*8..+7, channels chan(pass, c)
  auto chan = [&](int pass, int c) { return pass * PWD + tx * CPT + c; };

  // ---- upsample: Pj[16 pix][C] = f16(Wr . coarse) for the 4x4 coarse pixels under the window (K = 2C)
  if (up) {
    const int H2 = H >> 1, W2 = W >> 1, Hs = (H2 + 3) & ~3, Ws = (W2 + 3) & ~3, pr0 = r0 >> 1, pc0 = c0 >> 1;
    if (tid < C) {   // (pixel, 32-channel chunk of 2C)
      const int px = tid & 15, ch = tid >> 4;
      const int pr = pr0 + (px >> 2), pc = pc0 + (px & 3);
      uint32_t b[8] = {};
      if (pr >= 0 && pr < H2 && pc >= 0 && pc < W2) {
        const uint8_t* src = P.p30 + (size_t)(ch * 2) * Hs * Ws * 16 + (size_t)(pr * Ws + pc) * 16;
#pragma unroll
        for (int g = 0; g < 4; g++) { b[2 * g] = *(const uint32_t*)(src + 4 * g); b[2 * g + 1] = *(const uint32_t*)(src + (size_t)Hs * Ws * 16 + 4 * g); }
      }
#pragma unroll
      for (int g = 0; g < 4; g++)
#pragma unroll
        for (int s2 = 0; s2 < 2; s2++)
#pragma unroll
          for (int hh = 0; hh < 2; hh++) {
            int c = 16 * s2 + 4 * g + 2 * hh;
            R1[((ch * 32 + pi32(c)) >> 1) * 16 + px] = x256(e4x2((b[2 * g + s2] >> (16 * hh)) & 0xFFFFu));
          }
    }
    constexpr int CQ = C / 64;
    const int tx6 = tid & 63, ty6 = tid >> 6;
    float acc[4][CQ] = {};
#pragma unroll 1
    for (int kb = 0; kb < C / 16; kb++) {
      __syncthreads();
      stage(Xs, C, rec, C / 16, [&](int i, int& off, int& ncol, int& kpo) {
        off = R_AFT + (kb * (C / 16) + i) * 512; ncol = i * 16; kpo = 0; });
      __syncthreads();
#pragma unroll KPU
      for (int kp = 0; kp < 16; kp++) {
        uint4 a = *(const uint4*)&R1[(kb * 16 + kp) * 16 + ty6 * 4];
        uint32_t av[4] = {a.x, a.y, a.z, a.w}, bv[CQ];
        ldw<CQ>(&Xs[kp * C + tx6 * CQ], bv);
#pragma unroll
        for (int r = 0; r < 4; r++)
#pragma unroll
          for (int c = 0; c < CQ; c++) acc[r][c] = dot2(av[r], bv[c], acc[r][c]);
      }
    }
#pragma unroll
    for (int r = 0; r < 4; r++)
#pragma unroll
      for (int c = 0; c < CQ; c++) Pj[(ty6 * 4 + r) * C + tx6 * CQ + c] = (_Float16)(acc[r][c] * 256.f);
    __syncthreads();
  }

  // ---- load the window: token t = (qy*2+qx)*16 + y*4 + x
  for (int i = tid; i < 64 * (C / 32); i += 256) {   // (token, 32-channel chunk)
    int t = i & 63, ch = i >> 6;
    int row = r0 + ((t >> 5) & 1) * 4 + ((t >> 2) & 3), col = c0 + ((t >> 4) & 1) * 4 + (t & 3);
    bool in = row >= 0 && row < H && col >= 0 && col < W;
    uint32_t b[8] = {};
    if (in) {
      int p = (row & 3) * 4 + (col & 3);
      if (P.flags & 1) {
        const uint8_t* src = P.in + (size_t)(ch * 2) * H * W * 16 + (size_t)(row * W + col) * 16;
#pragma unroll
        for (int g = 0; g < 4; g++) { b[2 * g] = *(const uint32_t*)(src + 4 * g); b[2 * g + 1] = *(const uint32_t*)(src + (size_t)H * W * 16 + 4 * g); }
      } else {
        const uint8_t* src = P.in + (size_t)((row >> 2) * WT + (col >> 2)) * (C * 16) + ch * 512 + (p & 7) * 64 + 4 * (p >> 3);
#pragma unroll
        for (int g = 0; g < 4; g++) { b[2 * g] = *(const uint32_t*)(src + 16 * g); b[2 * g + 1] = *(const uint32_t*)(src + 16 * g + 8); }
      }
    }
#pragma unroll
    for (int g = 0; g < 4; g++)
#pragma unroll
      for (int s = 0; s < 2; s++)
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          int c = 16 * s + 4 * g + 2 * hh;
          uint32_t v = x256(e4x2((b[2 * g + s] >> (16 * hh)) & 0xFFFFu));
          const int rc = ch * 32 + pi32(c);
          if (up) {   // x = q8(f16(skip * gain + P(coarse))), zero outside the frame
            const int lr = ((t >> 5) & 1) * 4 + ((t >> 2) & 3), lc = ((t >> 4) & 1) * 4 + (t & 3);
            const int pix = (lr >> 1) * 4 + (lc >> 1);
            const _Float16* ug = (const _Float16*)(rec + R_RS + 2 * C + 32);
            float g0 = rc < 16 ? (float)rs[C + rc] : (float)ug[rc - 16];
            float g1 = rc + 1 < 16 ? (float)rs[C + 1 + rc] : (float)ug[rc - 15];
            float x0 = in ? q8(f16r((float)((double)lo(v) * g0 + (double)(float)Pj[pix * C + rc]))) : 0.f;
            float x1 = in ? q8(f16r((float)((double)hi(v) * g1 + (double)(float)Pj[pix * C + rc + 1]))) : 0.f;
            v = pk(x0, x1);
          }
          Xs[(rc >> 1) * 64 + t] = v;
        }
  }

  // I8: Xs becomes [x as e4m3 byte pairs, u16 [C/2][64]] | [Xq int8 quads, words [C/4][64]]
  uint16_t* Xb = (uint16_t*)Xs;
  uint32_t* Xq = Xs + C * 16;
  // Xs (f16, e4m3-grid values) -> Xb bytes + Xq int8 with a power-of-two scale sa_s[kb][t] per (token, 32-channel group kb)
  auto quant = [&]() {
    constexpr int NP = (64 * KB + 255) / 256;
    uint32_t q[NP][8], xb[NP][8];
    __syncthreads();
#pragma unroll
    for (int p = 0; p < NP; p++) {
      const int i = tid + 256 * p, t = i & 63, kb = i >> 6;
      if (i >= 64 * KB) break;
      uint32_t v[16];
      float m = 0.f;
#pragma unroll
      for (int j = 0; j < 16; j++) { v[j] = Xs[(kb * 16 + j) * 64 + t]; m = fmaxf(m, fmaxf(fabsf(lo(v[j])), fabsf(hi(v[j])))); }
      int e; frexpf(m, &e);   // m < 2^e: m * 2^(7-e) < 128
      const float sa = m > 0.f ? ldexpf(1.f, e - 7) : 1.f, inv = 1.f / sa;
      sa_s[kb * 64 + t] = sa;
#pragma unroll
      for (int j = 0; j < 8; j++) {
        float f[4] = {lo(v[2 * j]), hi(v[2 * j]), lo(v[2 * j + 1]), hi(v[2 * j + 1])};
        uint32_t w = 0;
#pragma unroll
        for (int l = 0; l < 4; l++) w |= ((uint32_t)(int)rintf(f[l] * inv) & 0xFFu) << (8 * l);
        q[p][j] = w;
        xb[p][j] = (to_e4m3(f[0]) | to_e4m3(f[1]) << 8) | (to_e4m3(f[2]) | to_e4m3(f[3]) << 8) << 16;
      }
    }
    __syncthreads();
#pragma unroll
    for (int p = 0; p < NP; p++) {
      const int i = tid + 256 * p, t = i & 63, kb = i >> 6;
      if (i >= 64 * KB) break;
#pragma unroll
      for (int j = 0; j < 8; j++) {
        Xq[(kb * 8 + j) * 64 + t] = q[p][j];
        Xb[(kb * 16 + 2 * j) * 64 + t] = (uint16_t)xb[p][j];
        Xb[(kb * 16 + 2 * j + 1) * 64 + t] = (uint16_t)(xb[p][j] >> 16);
      }
    }
  };
  if constexpr (I8) quant();

  float accc[NPASS][8][CPT];
#pragma unroll
  for (int a = 0; a < NPASS; a++)
#pragma unroll
    for (int r = 0; r < 8; r++)
#pragma unroll
      for (int c = 0; c < CPT; c++) accc[a][r][c] = 0.f;

  // ================= MLP, one head (128 hidden values) at a time, in NS sub-chunks of HC
  constexpr int CAP = C == 256 ? CAP256 : C == 128 ? CAP128 : 0;
  const int win = blockIdx.y * gridDim.x + blockIdx.x;
  const bool sp = SPLIT && win < CAP;
  uint32_t* gms = C == 256 ? g_ms256 : g_ms128;
  if (SPLIT && !sp && blockIdx.z) return;   // beyond capacity: z = 0 runs the window unsplit
  const int hb = sp ? (int)blockIdx.z * (NH / 2) : 0, he = sp ? hb + NH / 2 : NH;
  // int8 contract of head h: Ms int8 in R2 (+ scales at R2 + 512), weights [8 kq][C] words staged into R3
  auto con_i8 = [&](int h) {
    const uint4* src = (const uint4*)(P.scratch + OF::K + (size_t)h * 32 * C);
    __syncthreads();
    for (int i = tid; i < 2 * C; i += 256) ((uint4*)R3)[i] = src[i];
    __syncthreads();
    const float* msc = (const float*)R2 + 512;
#pragma unroll
    for (int pass = 0; pass < NPASS; pass++) {
      int32_t ai[8][CPT] = {};
#pragma unroll
      for (int kq = 0; kq < 8; kq++) {
        uint4 a0 = *(const uint4*)&R2[kq * 64 + ty * 8];
        uint4 a1 = *(const uint4*)&R2[kq * 64 + ty * 8 + 4];
        uint32_t av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w}, bv[CPT];
        ldw<CPT>(&R3[kq * C + chan(pass, 0)], bv);
#pragma unroll
        for (int r = 0; r < 8; r++)
#pragma unroll
          for (int c = 0; c < CPT; c++) ai[r][c] = __builtin_amdgcn_sdot4((int)av[r], (int)bv[c], ai[r][c], false);
      }
#pragma unroll
      for (int r = 0; r < 8; r++) {
        const float st = msc[ty * 8 + r];
#pragma unroll
        for (int c = 0; c < CPT; c++) accc[pass][r][c] = fmaf((float)ai[r][c], st, accc[pass][r][c]);
      }
    }
  };
#pragma unroll 1
  for (int h = SPLIT ? hb : 0; h < (SPLIT ? he : NH); h++) {
    const bool ldc = tid < PWD * 2;   // contract pieces of one pass
    auto con_src = [&](int pass) {
      return *((const uint4*)(rec + R_CON + (h * (C / 16) + pass * (PWD / 16)) * 512) + (ldc ? tid : 0)); };
    uint4 wc = con_src(0);
    float macc[2][4] = {};
    const int tx8 = tid & 7, ty8 = tid >> 3;
#pragma unroll 1
    for (int sc = 0; sc < NS; sc++) {
      // ---- expand: Hs[64 tok][HC] = q8(act(X . We[h*128 + sc*HC .. +HC]^T)), K = C (GN = 8)
      const bool ldw16 = tid < HC * 2;   // HC/16 tiles of 32 pieces
      uint4 wm = *((const uint4*)(rec + R_MID + (h * 8 + sc * (HC / 16)) * 512) + (ldw16 ? tid : 0));
      if constexpr (I8) {
        float acc[8][HCPT] = {};
        const uint4* wsrc = (const uint4*)(P.scratch + (size_t)((h * NS + sc) * KB) * HC * 32) + (ldw16 ? tid : 0);
        const float* sw = (const float*)(P.scratch + 4 * C * C) + h * 128 + sc * HC;
        uint4 wn = wsrc[0];
#pragma unroll 1
        for (int kb = 0; kb < KB; kb++) {
          __syncthreads();
          if (ldw16) ((uint4*)R3)[tid] = wn;
          __syncthreads();
          if (kb + 1 < KB) wn = wsrc[(kb + 1) * HC * 2];
          int32_t ai[8][HCPT] = {};
#pragma unroll
          for (int kq = 0; kq < 8; kq++) {
            uint4 a0 = *(const uint4*)&Xq[(kb * 8 + kq) * 64 + ty * 8];
            uint4 a1 = *(const uint4*)&Xq[(kb * 8 + kq) * 64 + ty * 8 + 4];
            uint32_t av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w}, bv[HCPT];
            ldw<HCPT>(&R3[kq * HC + tx * HCPT], bv);
#pragma unroll
            for (int r = 0; r < 8; r++)
#pragma unroll
              for (int c = 0; c < HCPT; c++) ai[r][c] = __builtin_amdgcn_sdot4((int)av[r], (int)bv[c], ai[r][c], false);
          }
#pragma unroll
          for (int r = 0; r < 8; r++) {
            const float st = sa_s[kb * 64 + ty * 8 + r];
#pragma unroll
            for (int c = 0; c < HCPT; c++) acc[r][c] = fmaf((float)ai[r][c], st, acc[r][c]);
          }
        }
        float swc[HCPT];
#pragma unroll
        for (int c = 0; c < HCPT; c++) swc[c] = sw[tx * HCPT + c];
        if constexpr (MID) {   // Hs -> int8 quads R1 [HC/4 kq][64 tok], scale per (token, 32-hidden group) in R2 + 768
          constexpr int LG = 32 / HCPT;   // lanes per group
          float* shs = (float*)R2 + 768;
#pragma unroll
          for (int r = 0; r < 8; r++) {
            float v[HCPT], m = 0.f;
#pragma unroll
            for (int c = 0; c < HCPT; c++) { v[c] = q8(act(acc[r][c] * swc[c])); m = fmaxf(m, fabsf(v[c])); }
#pragma unroll
            for (int o = 1; o < LG; o <<= 1) m = fmaxf(m, __shfl_xor(m, o));
            const float sa = pow2s(m), inv = 1.f / sa;
            const int t = ty * 8 + r, hid = tx * HCPT;
            if ((tx & (LG - 1)) == 0) shs[(hid >> 5) * 64 + t] = sa;
            uint32_t w = 0;
#pragma unroll
            for (int c = 0; c < HCPT; c++) w |= ((uint32_t)(int)rintf(v[c] * inv) & 0xFFu) << (8 * c);
            if constexpr (HCPT == 4) R1[(hid >> 2) * 64 + t] = w;
            else ((uint16_t*)R1)[((hid >> 2) * 64 + t) * 2 + ((hid >> 1) & 1)] = (uint16_t)w;
          }
        } else
#pragma unroll
        for (int r = 0; r < 8; r++)
#pragma unroll
          for (int c = 0; c < HCPT; c += 2)
            R1[((tx * HCPT + c) >> 1) * 64 + ty * 8 + r] = pk(q8(act(acc[r][c] * swc[c])), q8(act(acc[r][c + 1] * swc[c + 1])));
      } else {
        float acc[8][HCPT] = {};
        const uint4* wsrc = (const uint4*)(rec + R_EXP + (h * 8 * KB + sc * (HC / 16)) * 512) + (ldw16 ? tid : 0);
        uint4 wn = wsrc[0];
#pragma unroll 1
        for (int kb = 0; kb < KB; kb++) {
          __syncthreads();
          if (ldw16) put16(R3, HC, wn, tid & 31, (tid >> 5) * 16);
          __syncthreads();
          if (kb + 1 < KB) wn = wsrc[(kb + 1) * 256];
#pragma unroll KPU
          for (int kp = 0; kp < 16; kp++) {
            uint4 a0 = *(const uint4*)&Xs[(kb * 16 + kp) * 64 + ty * 8];
            uint4 a1 = *(const uint4*)&Xs[(kb * 16 + kp) * 64 + ty * 8 + 4];
            uint32_t av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w}, bv[HCPT];
            ldw<HCPT>(&R3[kp * HC + tx * HCPT], bv);
#pragma unroll
            for (int r = 0; r < 8; r++)
#pragma unroll
              for (int c = 0; c < HCPT; c++) acc[r][c] = dot2(av[r], bv[c], acc[r][c]);
          }
        }
#pragma unroll
        for (int r = 0; r < 8; r++)
#pragma unroll
          for (int c = 0; c < HCPT; c += 2)
            R1[((tx * HCPT + c) >> 1) * 64 + ty * 8 + r] = pk(q8(act(acc[r][c] * 256.f)), q8(act(acc[r][c + 1] * 256.f)));
      }
      __syncthreads();
      if constexpr (MID) {   // mid on int8: weights (h, sc) block [HC/4 kq][32 n] words from scratch
        if (ldw16) ((uint4*)R3)[tid] = ((const uint4*)(P.scratch + OF::M + (size_t)(h * NS + sc) * HC * 32))[tid];
        __syncthreads();
        const float* shs = (const float*)R2 + 768;
#pragma unroll
        for (int g = 0; g < HC / 32; g++) {
          int32_t ai[2][4] = {};
#pragma unroll
          for (int kq = 0; kq < 8; kq++) {
            uint2 a = *(const uint2*)&R1[(g * 8 + kq) * 64 + ty8 * 2];
            uint4 bb = *(const uint4*)&R3[(g * 8 + kq) * 32 + tx8 * 4];
            uint32_t av[2] = {a.x, a.y}, bv[4] = {bb.x, bb.y, bb.z, bb.w};
#pragma unroll
            for (int r = 0; r < 2; r++)
#pragma unroll
              for (int c = 0; c < 4; c++) ai[r][c] = __builtin_amdgcn_sdot4((int)av[r], (int)bv[c], ai[r][c], false);
          }
#pragma unroll
          for (int r = 0; r < 2; r++) {
            const float st = shs[g * 64 + ty8 * 2 + r];
#pragma unroll
            for (int c = 0; c < 4; c++) macc[r][c] = fmaf((float)ai[r][c], st, macc[r][c]);
          }
        }
        (void)wm;
        continue;
      }
      // ---- mid partial: macc += Hs . Wm_h[:, sc*HC .. +HC]^T (GN = 2); local tile i = tid>>5 -> kb = i>>1, j = i&1
      if (ldw16) put16(R3 + (tid >> 6) * 16 * 32, 32, wm, tid & 31, ((tid >> 5) & 1) * 16);
      __syncthreads();
#pragma unroll 8
      for (int kp = 0; kp < HC / 2; kp++) {
        uint2 a = *(const uint2*)&R1[kp * 64 + ty8 * 2];
        uint4 bb = *(const uint4*)&R3[kp * 32 + tx8 * 4];
        uint32_t av[2] = {a.x, a.y}, bv[4] = {bb.x, bb.y, bb.z, bb.w};
#pragma unroll
        for (int r = 0; r < 2; r++)
#pragma unroll
          for (int c = 0; c < 4; c++) macc[r][c] = dot2(av[r], bv[c], macc[r][c]);
      }
    }
    // Ms[64 tok][32] = q8(f16(mid)); the previous head's contract finished reading R2 long ago
    float msw[4];   // mid output scale (int8 mid: per output channel; f16: 2^8)
#pragma unroll
    for (int c = 0; c < 4; c++) msw[c] = MID ? ((const float*)(P.scratch + OF::M + 128 * C))[h * 32 + tx8 * 4 + c] : 256.f;
    if constexpr (CON) {   // Ms -> int8 quads R2 [8 kq][64 tok] + scale per token at R2 + 512
#pragma unroll
      for (int r = 0; r < 2; r++) {
        float v[4], m = 0.f;
#pragma unroll
        for (int c = 0; c < 4; c++) { v[c] = q8(f16r(macc[r][c] * msw[c])); m = fmaxf(m, fabsf(v[c])); }
#pragma unroll
        for (int o = 1; o < 8; o <<= 1) m = fmaxf(m, __shfl_xor(m, o));
        const float sa = pow2s(m), inv = 1.f / sa;
        uint32_t w = 0;
#pragma unroll
        for (int c = 0; c < 4; c++) w |= ((uint32_t)(int)rintf(v[c] * inv) & 0xFFu) << (8 * c);
        R2[tx8 * 64 + ty8 * 2 + r] = w;
        if (tx8 == 0) ((float*)R2)[512 + ty8 * 2 + r] = sa;
      }
    } else
#pragma unroll
    for (int r = 0; r < 2; r++)
#pragma unroll
      for (int c = 0; c < 4; c += 2)
        R2[((tx8 * 4 + c) >> 1) * 64 + ty8 * 2 + r] = pk(q8(f16r(macc[r][c] * msw[c])), q8(f16r(macc[r][c + 1] * msw[c + 1])));
    if (SPLIT && sp && CON) {   // int8 Ms: 512 words + 64 scales, written by other threads
      __syncthreads();
      for (int i = tid; i < 576; i += 256) gms[(win * NH + h) * 1024 + i] = R2[i];
      continue;
    }
    if (SPLIT && sp) {   // hand Ms to the contract workgroup
#pragma unroll
      for (int r = 0; r < 2; r++)
#pragma unroll
        for (int c = 0; c < 4; c += 2) {
          const int i = ((tx8 * 4 + c) >> 1) * 64 + ty8 * 2 + r;
          gms[(win * NH + h) * 1024 + i] = R2[i];
        }
      continue;
    }
    // ---- contract partial: accc += Ms . Wc[:, h*32 .. +32]^T (GN = C/16), passes of PWD output channels
    if constexpr (CON) { con_i8(h); continue; }
#pragma unroll
    for (int pass = 0; pass < NPASS; pass++) {
      __syncthreads();
      if (ldc) put16(R3, PWD, wc, tid & 31, (tid >> 5) * 16);
      __syncthreads();
      if (pass + 1 < NPASS) wc = con_src(pass + 1);
#pragma unroll KPU
      for (int kp = 0; kp < 16; kp++) {
        uint4 a0 = *(const uint4*)&R2[kp * 64 + ty * 8];
        uint4 a1 = *(const uint4*)&R2[kp * 64 + ty * 8 + 4];
        uint32_t av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w}, bv[CPT];
        ldw<CPT>(&R3[kp * PWD + tx * CPT], bv);
#pragma unroll
        for (int r = 0; r < 8; r++)
#pragma unroll
          for (int c = 0; c < CPT; c++) accc[pass][r][c] = dot2(av[r], bv[c], accc[pass][r][c]);
      }
    }
  }

  if constexpr (SPLIT) if (sp) {   // the window's second workgroup to arrive runs the contract for all heads, in order
    __shared__ uint32_t s_last;
    uint32_t* cnt = (C == 256 ? g_cnt256 : g_cnt128) + win;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    if (tid == 0) {
      s_last = __hip_atomic_fetch_add(cnt, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
      if (s_last) __hip_atomic_store(cnt, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);   // reset for the next dispatch
    }
    __syncthreads();
    if (!s_last) return;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    const bool ldc = tid < PWD * 2;
#pragma unroll 1
    for (int h = 0; h < NH; h++) {   // the contract code above, Ms from g_ms
      if constexpr (CON) {
        __syncthreads();
        for (int i = tid; i < 576; i += 256) R2[i] = gms[(win * NH + h) * 1024 + i];
        con_i8(h);
        continue;
      }
      auto con_src = [&](int pass) {
        return *((const uint4*)(rec + R_CON + (h * (C / 16) + pass * (PWD / 16)) * 512) + (ldc ? tid : 0)); };
      uint4 wc = con_src(0);
      __syncthreads();
#pragma unroll
      for (int k = 0; k < 4; k++) R2[tid + 256 * k] = gms[(win * NH + h) * 1024 + tid + 256 * k];
#pragma unroll
      for (int pass = 0; pass < NPASS; pass++) {
        __syncthreads();
        if (ldc) put16(R3, PWD, wc, tid & 31, (tid >> 5) * 16);
        __syncthreads();
        if (pass + 1 < NPASS) wc = con_src(pass + 1);
#pragma unroll KPU
        for (int kp = 0; kp < 16; kp++) {
          uint4 a0 = *(const uint4*)&R2[kp * 64 + ty * 8];
          uint4 a1 = *(const uint4*)&R2[kp * 64 + ty * 8 + 4];
          uint32_t av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w}, bv[CPT];
          ldw<CPT>(&R3[kp * PWD + tx * CPT], bv);
#pragma unroll
          for (int r = 0; r < 8; r++)
#pragma unroll
            for (int c = 0; c < CPT; c++) accc[pass][r][c] = dot2(av[r], bv[c], accc[pass][r][c]);
        }
      }
    }
  }

  // ---- y = rs * x + mlp ; Yq = q8(f16(y)) replaces x in Xs (multi-head: it is both attention input and skip)
  {
    uint32_t yq[NPASS][8][CPT / 2];
#pragma unroll
    for (int pass = 0; pass < NPASS; pass++)
#pragma unroll
      for (int r = 0; r < 8; r++)
#pragma unroll
        for (int c = 0; c < CPT; c += 2) {
          int ch = chan(pass, c), t = ty * 8 + r;
          uint32_t xv = I8 ? x256(e4x2(Xb[(ch >> 1) * 64 + t])) : Xs[(ch >> 1) * 64 + t];
          const float* swk = (const float*)(P.scratch + OF::K + C * C);
          float y0 = __fadd_rn(__fmul_rn(lo(xv), (float)rs[rso + ch]), accc[pass][r][c] * (CON ? swk[ch] : 256.f));
          float y1 = __fadd_rn(__fmul_rn(hi(xv), (float)rs[rso + ch + 1]), accc[pass][r][c + 1] * (CON ? swk[ch + 1] : 256.f));
          yq[pass][r][c >> 1] = pk(q8(f16r(y0)), q8(f16r(y1)));
        }
    __syncthreads();
#pragma unroll
    for (int pass = 0; pass < NPASS; pass++)
#pragma unroll
      for (int r = 0; r < 8; r++)
#pragma unroll
        for (int c = 0; c < CPT / 2; c++) Xs[(chan(pass, 0) / 2 + c) * 64 + ty * 8 + r] = yq[pass][r][c];
    if constexpr (I8) { quant(); __syncthreads(); }
  }

  // ================= attention, one head at a time; out_proj accumulated per head
  float acco[NPASS][8][CPT];
#pragma unroll
  for (int a = 0; a < NPASS; a++)
#pragma unroll
    for (int r = 0; r < 8; r++)
#pragma unroll
      for (int c = 0; c < CPT; c++) acco[a][r][c] = 0.f;
  // buffers per head; SMALL moves k, kn, Pm into R3 and qn, ctx, Wo, the reciprocals into R1
  float* qb = (float*)R1;
  float* kbuf = SMALL ? (float*)R3 : (float*)R1 + 2048;
  uint32_t* qn = SMALL ? R1 : R3;
  uint32_t* kn = SMALL ? R3 : R3 + 1024;
  uint32_t* Pm = SMALL ? R3 : R1;
  uint32_t* ctx = SMALL ? R1 : R3;
  uint32_t* wo = SMALL ? R1 + 1024 : R1;
  _Float16* Vt = (_Float16*)R2;
  const float* hsc = (const float*)(rec + R_HS);

#pragma unroll 1
  for (int h = 0; h < NH; h++) {
    // ---- qkv_h: [64 tok][96] = Yq . Wqkv[h*96 .. +96]^T, K = C (GN = 3C/16)
    {
      const int tx16 = tid & 15, ty16 = tid >> 4;
      float acc[4][6] = {};
      const bool ld = tid < 192;
      if constexpr (I8) {   // int8 copy: (h, kb) blocks of [8 kq][96] words, then float scales[3C]
        const uint8_t* q0 = P.scratch + 4 * C * C + 16 * C;
        const uint4* wsrc = (const uint4*)(q0 + (size_t)h * KB * 3072) + (ld ? tid : 0);
        uint4 wn = wsrc[0];
#pragma unroll 1
        for (int kb = 0; kb < KB; kb++) {
          __syncthreads();
          if (ld) ((uint4*)R3)[tid] = wn;
          __syncthreads();
          if (kb + 1 < KB) wn = wsrc[(kb + 1) * 192];
          int32_t ai[4][6] = {};
#pragma unroll
          for (int kq = 0; kq < 8; kq++) {
            uint4 a = *(const uint4*)&Xq[(kb * 8 + kq) * 64 + ty16 * 4];
            uint2 b0 = *(const uint2*)&R3[kq * 96 + tx16 * 6];
            uint2 b1 = *(const uint2*)&R3[kq * 96 + tx16 * 6 + 2];
            uint2 b2 = *(const uint2*)&R3[kq * 96 + tx16 * 6 + 4];
            uint32_t av[4] = {a.x, a.y, a.z, a.w}, bv[6] = {b0.x, b0.y, b1.x, b1.y, b2.x, b2.y};
#pragma unroll
            for (int r = 0; r < 4; r++)
#pragma unroll
              for (int c = 0; c < 6; c++) ai[r][c] = __builtin_amdgcn_sdot4((int)av[r], (int)bv[c], ai[r][c], false);
          }
#pragma unroll
          for (int r = 0; r < 4; r++) {
            const float st = sa_s[kb * 64 + ty16 * 4 + r];
#pragma unroll
            for (int c = 0; c < 6; c++) acc[r][c] = fmaf((float)ai[r][c], st, acc[r][c]);
          }
        }
        const float* sw = (const float*)(q0 + 3 * C * C) + h * 96 + tx16 * 6;
#pragma unroll
        for (int r = 0; r < 4; r++)
#pragma unroll
          for (int c = 0; c < 6; c++) acc[r][c] *= sw[c] * (1.f / 256.f);   // the epilogue below multiplies by 256
      } else {
      const uint4* wsrc = (const uint4*)(rec + R_QKV + h * 6 * 512) + (ld ? tid : 0);
      uint4 wn = wsrc[0];
#pragma unroll 1
      for (int kb = 0; kb < KB; kb++) {
        __syncthreads();
        if (ld) put16(R3, 96, wn, tid & 31, (tid >> 5) * 16);
        __syncthreads();
        if (kb + 1 < KB) wn = wsrc[(kb + 1) * (3 * C / 16) * 32];
#pragma unroll KPU
        for (int kp = 0; kp < 16; kp++) {
          uint4 a = *(const uint4*)&Xs[(kb * 16 + kp) * 64 + ty16 * 4];
          uint2 b0 = *(const uint2*)&R3[kp * 96 + tx16 * 6];
          uint2 b1 = *(const uint2*)&R3[kp * 96 + tx16 * 6 + 2];
          uint2 b2 = *(const uint2*)&R3[kp * 96 + tx16 * 6 + 4];
          uint32_t av[4] = {a.x, a.y, a.z, a.w}, bv[6] = {b0.x, b0.y, b1.x, b1.y, b2.x, b2.y};
#pragma unroll
          for (int r = 0; r < 4; r++)
#pragma unroll
            for (int c = 0; c < 6; c++) acc[r][c] = dot2(av[r], bv[c], acc[r][c]);
        }
      }
      }
      __syncthreads();   // everyone is done with R1 (Wo of the previous head) and R2 (Vt)
#pragma unroll
      for (int r = 0; r < 4; r++)
#pragma unroll
        for (int c = 0; c < 6; c++) {
          int n = tx16 * 6 + c, t = ty16 * 4 + r;
          float v = acc[r][c] * 256.f;
          if (n < 32) qb[n * 64 + t] = v;
          else if (n < 64) kbuf[(n - 32) * 64 + t] = v;
          else Vt[(n - 64) * 64 + t] = (_Float16)q8(f16r(v));
        }
    }
    __syncthreads();
    // ---- L2-normalise q (with the head scale) and k, q8 -> R3 qn [16 kp][64], kn [16 kp][64]
    if (tid < 128) {
      const int t = tid & 63, w = tid >> 6;
      float v[32];
#pragma unroll
      for (int d = 0; d < 32; d++) v[d] = (w ? kbuf : qb)[d * 64 + t];
      float sc = norm_scale(v);
      if (w == 0) sc = __fmul_rn(sc, hsc[h]);
#pragma unroll
      for (int d = 0; d < 32; d += 2) (w ? kn : qn)[(d >> 1) * 64 + t] = pk(q8(__fmul_rn(v[d], sc)), q8(__fmul_rn(v[d + 1], sc)));
    }
    __syncthreads();
    // ---- logits -> exponent-ramp numerators, Pm[key pair][query] (f16 pairs)
    {
      const int t = tid & 63, kq = tid >> 6;
      uint32_t qv[16];
#pragma unroll
      for (int i = 0; i < 16; i++) qv[i] = qn[i * 64 + t];
      const _Float16* braw = (const _Float16*)(rec + R_BIAS) + h * 4096;
      uint32_t num[8];
#pragma unroll
      for (int jj = 0; jj < 8; jj++) {
        float s2[2];
#pragma unroll
        for (int e = 0; e < 2; e++) {
          int s = kq * 16 + jj * 2 + e;
          float acc = 0.f;
#pragma unroll
          for (int i = 0; i < 16; i++) acc = dot2(qv[i], kn[i * 64 + s], acc);
          int ii = t & 15, j = s & 15;
          int lane = 4 * (ii & 7) + ((j & 7) >> 1);
          int slot = (j & 1) | ((ii >> 3) << 1) | ((j >> 3) << 2);
          float b = (float)braw[(4 * (t >> 4) + (s >> 4)) * 256 + lane * 8 + slot];
          s2[e] = fmaf(acc, 0.044921875f, fmaf(b, 0.044921875f, 1.30078125f));
        }
        h2 r = (h2){(_Float16)s2[0], (_Float16)s2[1]};
        r = __builtin_elementwise_max(r, (h2){(_Float16)1.03125f, (_Float16)1.03125f});
        r = __builtin_elementwise_min(r, (h2){(_Float16)1.5693359375f, (_Float16)1.5693359375f});
        num[jj] = (__builtin_bit_cast(uint32_t, r) & 0x03FF03FFu) << 5;
      }
      if constexpr (SMALL) __syncthreads();   // SMALL: Pm replaces kn (else Pm overlays q f32, last read before the previous barrier)
#pragma unroll
      for (int jj = 0; jj < 8; jj++) Pm[(kq * 8 + jj) * 64 + t] = num[jj];
    }
    __syncthreads();
    // ---- denominator in the reference order (as native-swin32), one thread per query
    if (tid < 64) {
      h2 pp[32];
#pragma unroll
      for (int j = 0; j < 32; j++) pp[j] = __builtin_bit_cast(h2, Pm[j * 64 + tid]);
      h2 hm[2];
#pragma unroll
      for (int m = 0; m < 2; m++) {
        h2 pr[4];
#pragma unroll
        for (int p = 0; p < 4; p++) { int b = 2 * p + m; pr[p] = ((pp[b] + pp[8 + b]) + pp[16 + b]) + pp[24 + b]; }
        hm[m] = ((pr[0] + pr[1]) + pr[2]) + pr[3];
      }
      _Float16 den = (hm[0].x + hm[0].y) + (hm[1].x + hm[1].y);
      rec_s[tid] = __fdiv_rn(1.f, fmaxf((float)den, 0.00006103515625f));
    }
    __syncthreads();
    // Normalise each P pair once for all four output-channel groups.
    // Pm holds 32 * 64 pairs; each thread owns eight disjoint pairs.
    // Denominator readers finished at the preceding workgroup barrier.
    {
      const int t = tid & 63, group = tid >> 6;
      const float rc = rec_s[t];
#pragma unroll
      for (int n = 0; n < 8; n++) {
        const int j = group * 8 + n;
        const uint32_t u = Pm[j * 64 + t];
        // q8(f16r()) of both halves at once, on the f16 grid (q8h from pre/post_native, proof q8h_cpu.py)
        const uint32_t xb = pk(__fmul_rn(lo(u), rc), __fmul_rn(hi(u), rc));
#ifdef Q8FAST
        Pm[j * 64 + t] = (xb + 0x00400040u) & 0xFF80FF80u;
#else
        typedef unsigned short u2 __attribute__((ext_vector_type(2)));
        const u2 ab = __builtin_elementwise_min(__builtin_bit_cast(u2, xb & 0x7FFF7FFFu), (u2){0x5F00, 0x5F00});
        const u2 eb = __builtin_elementwise_max(ab & (u2){0x7C00, 0x7C00}, (u2){0x2400, 0x2400});
        const h2 K = __builtin_bit_cast(h2, eb) * (h2){(_Float16)128.f, (_Float16)128.f};
        const h2 r = (__builtin_bit_cast(h2, ab) + K) - K;
        Pm[j * 64 + t] = (__builtin_bit_cast(uint32_t, r) & 0x7FFF7FFFu) | (xb & 0x80008000u);
#endif
      }
    }
    __syncthreads(); // all channel groups consume all 32 normalised pairs
    // ---- context = P V, q8(f16()) -> R3 ctx [16 kp][64 tok]
    {
      const int t = tid & 63, dq = tid >> 6;
      float s[8] = {};
#pragma unroll 8
      for (int j = 0; j < 32; j++) {
        uint32_t pv = Pm[j * 64 + t];
#pragma unroll
        for (int d = 0; d < 8; d++) s[d] = dot2(pv, ((const uint32_t*)Vt)[(dq * 8 + d) * 32 + j], s[d]);
      }
      __syncthreads();
      if constexpr (OUTP) {   // ctx -> int8 quads [8 kq][64 tok], scale per token (max over the 4 dq threads via cmx)
        float* cmx = SMALL ? (float*)R3 : (float*)R1 + 3072;
        float v[8], m = 0.f;
#pragma unroll
        for (int d = 0; d < 8; d++) { v[d] = q8(f16r(s[d])); m = fmaxf(m, fabsf(v[d])); }
        cmx[dq * 64 + t] = m;
        __syncthreads();
        m = fmaxf(fmaxf(cmx[t], cmx[64 + t]), fmaxf(cmx[128 + t], cmx[192 + t]));
        const float sa = pow2s(m), inv = 1.f / sa;
        if (dq == 0) cmx[256 + t] = sa;
#pragma unroll
        for (int j = 0; j < 2; j++) {
          uint32_t w = 0;
#pragma unroll
          for (int c = 0; c < 4; c++) w |= ((uint32_t)(int)rintf(v[4 * j + c] * inv) & 0xFFu) << (8 * c);
          ctx[(dq * 2 + j) * 64 + t] = w;
        }
      } else
#pragma unroll
      for (int d = 0; d < 8; d += 2) ctx[((dq * 8 + d) >> 1) * 64 + t] = pk(q8(f16r(s[d])), q8(f16r(s[d + 1])));
    }
    if constexpr (OUTP) {   // out_proj on int8: Wo block [8 kq][C] words from scratch into wo
      const uint4* src = (const uint4*)(P.scratch + OF::O + (size_t)h * 32 * C);
      for (int i = tid; i < 2 * C; i += 256) ((uint4*)wo)[i] = src[i];
      __syncthreads();
      const float* so = (SMALL ? (const float*)R3 : (const float*)R1 + 3072) + 256;
#pragma unroll
      for (int pass = 0; pass < NPASS; pass++) {
        int32_t ai[8][CPT] = {};
#pragma unroll
        for (int kq = 0; kq < 8; kq++) {
          uint4 a0 = *(const uint4*)&ctx[kq * 64 + ty * 8];
          uint4 a1 = *(const uint4*)&ctx[kq * 64 + ty * 8 + 4];
          uint32_t av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w}, bv[CPT];
          ldw<CPT>(&wo[kq * C + chan(pass, 0)], bv);
#pragma unroll
          for (int r = 0; r < 8; r++)
#pragma unroll
            for (int c = 0; c < CPT; c++) ai[r][c] = __builtin_amdgcn_sdot4((int)av[r], (int)bv[c], ai[r][c], false);
        }
#pragma unroll
        for (int r = 0; r < 8; r++) {
          const float st = so[ty * 8 + r];
#pragma unroll
          for (int c = 0; c < CPT; c++) acco[pass][r][c] = fmaf((float)ai[r][c], st, acco[pass][r][c]);
        }
      }
      continue;
    }
    // ---- out_proj partial: acco += ctx_h . Wo[:, h*32 .. +32]^T, Wo staged into R1 [16 kp][C] (GN = C/16)
    stage(wo, C, rec, C / 16, [&](int i, int& off, int& ncol, int& kpo) {
      off = R_OUT + (h * (C / 16) + i) * 512; ncol = i * 16; kpo = 0; });
    __syncthreads();
#pragma unroll KPU
    for (int kp = 0; kp < 16; kp++) {
      uint4 a0 = *(const uint4*)&ctx[kp * 64 + ty * 8];
      uint4 a1 = *(const uint4*)&ctx[kp * 64 + ty * 8 + 4];
      uint32_t av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
      for (int pass = 0; pass < NPASS; pass++) {
        uint32_t bv[CPT];
        ldw<CPT>(&wo[kp * C + chan(pass, 0)], bv);
#pragma unroll
        for (int r = 0; r < 8; r++)
#pragma unroll
          for (int c = 0; c < CPT; c++) acco[pass][r][c] = dot2(av[r], bv[c], acco[pass][r][c]);
      }
    }
  }

  // ---- out = ars * skip + Wo . ctx ; wide = f16(out) ; store q8(wide) as e4m3 in the arena layout
  {
    const _Float16* ars = (const _Float16*)(rec + R_ARS);
#pragma unroll
    for (int r = 0; r < 8; r++) {
      int t = ty * 8 + r;
      int row = r0 + ((t >> 5) & 1) * 4 + ((t >> 2) & 3), col = c0 + ((t >> 4) & 1) * 4 + (t & 3);
      bool in = row >= 0 && row < H && col >= 0 && col < W;
      int p = (row & 3) * 4 + (col & 3);
      uint8_t* dst = P.out + (size_t)((row >> 2) * WT + (col >> 2)) * (C * 16) + (p & 7) * 64 + 4 * (p >> 3);
#pragma unroll
      for (int pass = 0; pass < NPASS; pass++)
#pragma unroll
        for (int c = 0; c < CPT; c++) {
          int ch = chan(pass, c);
          uint32_t sk = I8 ? x256(e4x2(Xb[(ch >> 1) * 64 + t])) : Xs[(ch >> 1) * 64 + t];
          float o = __fadd_rn(__fmul_rn((ch & 1) ? hi(sk) : lo(sk), (float)ars[ch]),
                              acco[pass][r][c] * (OUTP ? ((const float*)(P.scratch + OF::O + C * C))[ch] : 256.f));
          acco[pass][r][c] = o;
          if (!in) continue;
          int mc = ipi32(ch & 31);
          uint8_t v8 = (uint8_t)to_e4m3(q8(f16r(o)));
          if (P.flags & 2)
            P.out[(size_t)((ch >> 5) * 2 + (mc >> 4)) * H * W * 16 + (size_t)(row * W + col) * 16 + (mc & 15)] = v8;
          else
            dst[(ch >> 5) * 512 + 16 * ((mc & 15) >> 2) + 8 * (mc >> 4) + (mc & 3)] = v8;
        }
    }
  }
  if (!(P.flags & 4)) return;

  // ---- downsample: pooled = q8(pool2x2(out)), next = q8(f16(Wds . pooled)) -> +0x38, planes [2C/16][H/2][W/2][16].
  //      A thread's 8 tokens are 2 rows x 4 columns = 2 pooled pixels.
  __syncthreads();   // R1 (Wo) and Xs (skip) are free from here on
  {
    const int py = (ty >> 2) * 2 + (ty & 1), qx = (ty >> 1) & 1;
#pragma unroll
    for (int px = 0; px < 2; px++)
#pragma unroll
      for (int pass = 0; pass < NPASS; pass++)
#pragma unroll
        for (int c = 0; c < CPT; c += 2) {
          float v[2];
#pragma unroll
          for (int e = 0; e < 2; e++) {
            _Float16 h0 = (_Float16)(acco[pass][2 * px][c + e] + acco[pass][2 * px + 1][c + e]);
            _Float16 h1 = (_Float16)(acco[pass][4 + 2 * px][c + e] + acco[pass][4 + 2 * px + 1][c + e]);
            _Float16 tot = h0 + h1;
            v[e] = q8((float)tot * 0.25f);
          }
          R1[(chan(pass, c) >> 1) * 16 + py * 4 + qx * 2 + px] = pk(v[0], v[1]);
        }
  }
  {
    constexpr int CP2 = C / 32;   // 2C outputs over 64 threads
    const int tx6 = tid & 63, ty6 = tid >> 6;
    float acc[4][CP2] = {};
#pragma unroll 1
    for (int kb = 0; kb < KB; kb++) {
      __syncthreads();
      stage(Xs, 2 * C, rec, C / 8, [&](int i, int& off, int& ncol, int& kpo) {
        off = R_DSRS + (kb * (C / 8) + i) * 512; ncol = i * 16; kpo = 0; });
      __syncthreads();
#pragma unroll KPU
      for (int kp = 0; kp < 16; kp++) {
        uint4 a = *(const uint4*)&R1[(kb * 16 + kp) * 16 + ty6 * 4];
        uint32_t av[4] = {a.x, a.y, a.z, a.w}, bv[CP2];
        ldw<CP2>(&Xs[kp * 2 * C + tx6 * CP2], bv);
#pragma unroll
        for (int r = 0; r < 4; r++)
#pragma unroll
          for (int c = 0; c < CP2; c++) acc[r][c] = dot2(av[r], bv[c], acc[r][c]);
      }
    }
    const int H2 = H >> 1, W2 = W >> 1, Hs = (H2 + 3) & ~3, Ws = (W2 + 3) & ~3;
#pragma unroll
    for (int r = 0; r < 4; r++) {
      int pix = ty6 * 4 + r, pr = (r0 >> 1) + (pix >> 2), pc = (c0 >> 1) + (pix & 3);
      if (pr < 0 || pr >= H2 || pc < 0 || pc >= W2) continue;
#pragma unroll
      for (int c = 0; c < CP2; c++) {
        int ch = tx6 * CP2 + c, mc = ipi32(ch & 31);
        P.p38[(size_t)((ch >> 5) * 2 + (mc >> 4)) * Hs * Ws * 16 + (size_t)(pr * Ws + pc) * 16 + (mc & 15)] =
            (uint8_t)to_e4m3(q8(f16r(acc[r][c] * 256.f)));
      }
    }
  }
}

extern "C" __global__ __launch_bounds__(256) void k_swin_c256(VarParams P) { swin_block<256>(P); }
extern "C" __global__ __launch_bounds__(256) void k_swin_c128(VarParams P) { swin_block<128>(P); }
// split variants, launch with grid z = 2 (see SPLIT above)
extern "C" __global__ __launch_bounds__(256) void k_swin_c256s(VarParams P) { swin_block<256, true>(P); }
extern "C" __global__ __launch_bounds__(256) void k_swin_c128s(VarParams P) { swin_block<128, true>(P); }
extern "C" __global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(8))) void k_swin_c64(VarParams P) { swin_block<64>(P); }

// ---- int8_swin prep (once per record, launched by the runtime at load with the step's kernarg, P.scratch = destination,
// at least 4*C*C + 16*C bytes): expand weights [4C][C] -> int8, power-of-two scale per output channel, LDS-ready order
// (h, sc, kb) blocks of [8 kq][HC] words; then float scales[4C].
template <int C>
__device__ void i8_prep(const VarParams& P) {
  constexpr int HC = C == 64 ? 64 : 128, NS = 128 / HC, KB = C / 32;
  int8_t* dst = (int8_t*)P.scratch;
  float* sc = (float*)(P.scratch + 4 * C * C);
  for (int n = blockIdx.x * 256 + threadIdx.x; n < 4 * C; n += gridDim.x * 256) {
    auto w = [&](int k) {
      int tile = (n >> 7) * (8 * C / 32) + (k >> 5) * 8 + ((n & 127) >> 4), nn = n & 15, kk = k & 31;
      uint8_t b = P.w[tile * 512 + (nn & 7) * 64 + ((kk >> 1) & 3) * 16 + (nn >> 3) * 8 + (kk >> 3) * 2 + (kk & 1)];
      return lo(e4x2(b)) * 256.f;
    };
    float m = 0.f;
    for (int k = 0; k < C; k++) m = fmaxf(m, fabsf(w(k)));
    int e; frexpf(m, &e);
    const float s = m > 0.f ? ldexpf(1.f, e - 7) : 1.f;
    sc[n] = s;
    const int h = n >> 7, s_ = (n & 127) / HC, nn = n % HC;
    for (int k = 0; k < C; k++)
      dst[(size_t)((h * NS + s_) * KB + (k >> 5)) * HC * 32 + ((((k & 31) >> 2) * HC + nn) << 2) + (k & 3)] = (int8_t)rintf(w(k) / s);
  }
}
template <int C>
__device__ void i8_prep_qkv(const VarParams& P) {
  constexpr int KB = C / 32, GN = 3 * C / 16;
  const bool up = P.flags & 8;
  constexpr int R_AFT = 4 * C * C + C * 128 + C * C;
  const int R_RS = up ? R_AFT + 2 * C * C : R_AFT, R_QKV = R_RS + 2 * C + 32 + (up ? 2 * C - 32 : 0);
  int8_t* dst = (int8_t*)P.scratch + 4 * C * C + 16 * C;
  float* sc = (float*)(dst + 3 * C * C);
  for (int n = blockIdx.x * 256 + threadIdx.x; n < 3 * C; n += gridDim.x * 256) {
    auto w = [&](int k) {
      int tile = (k >> 5) * GN + (n >> 4), nn = n & 15, kk = k & 31;
      uint8_t b = P.w[R_QKV + tile * 512 + (nn & 7) * 64 + ((kk >> 1) & 3) * 16 + (nn >> 3) * 8 + (kk >> 3) * 2 + (kk & 1)];
      return lo(e4x2(b)) * 256.f;
    };
    float m = 0.f;
    for (int k = 0; k < C; k++) m = fmaxf(m, fabsf(w(k)));
    int e; frexpf(m, &e);
    const float s = m > 0.f ? ldexpf(1.f, e - 7) : 1.f;
    sc[n] = s;
    const int h = n / 96, nn = n % 96;
    for (int k = 0; k < C; k++)
      dst[(size_t)(h * KB + (k >> 5)) * 3072 + ((((k & 31) >> 2) * 96 + nn) << 2) + (k & 3)] = (int8_t)rintf(w(k) / s);
  }
}
// generic prep: [N][K] e4m3 weight matrix in the 16x32 tile layout (GN n-tiles per group) -> int8 at dst[di(n, k)],
// power-of-two scale per output n in sc[n]
__device__ __forceinline__ float wtile(const uint8_t* m, int GN, int K, int n, int k) {
  const int tile = (n / (16 * GN)) * (GN * K / 32) + (k >> 5) * GN + ((n % (16 * GN)) >> 4), nn = n & 15, kk = k & 31;
  return lo(e4x2(m[tile * 512 + (nn & 7) * 64 + ((kk >> 1) & 3) * 16 + (nn >> 3) * 8 + (kk >> 3) * 2 + (kk & 1)])) * 256.f;
}
template <typename D>
__device__ void prep_mat(const uint8_t* m, int N, int K, int GN, int8_t* dst, float* sc, D di) {
  for (int n = blockIdx.x * 256 + threadIdx.x; n < N; n += gridDim.x * 256) {
    float mx = 0.f;
    for (int k = 0; k < K; k++) mx = fmaxf(mx, fabsf(wtile(m, GN, K, n, k)));
    const float s = pow2s(mx);
    sc[n] = s;
    for (int k = 0; k < K; k++) dst[di(n, k)] = (int8_t)rintf(wtile(m, GN, K, n, k) / s);
  }
}
// mid, contract, out_proj (mask bits 1..3); record offsets as in swin_block
template <int C>
__device__ void i8_prep_rest(const VarParams& P) {
  constexpr int NH = C / 32, HC = C == 64 ? 64 : 128, NS = 128 / HC;
  using OF = I8Off<C>;
  const bool up = P.flags & 8;
  constexpr int R_MID = 4 * C * C, R_CON = R_MID + C * 128, R_AFT = R_CON + C * C;
  const int R_RS = up ? R_AFT + 2 * C * C : R_AFT, R_QKV = R_RS + 2 * C + 32 + (up ? 2 * C - 32 : 0);
  const int R_BIAS = R_QKV + 3 * C * C, R_HS = R_BIAS + NH * 8192, R_OUT = R_HS + (4 * NH > 16 ? 4 * NH : 16);
  int8_t* s8 = (int8_t*)P.scratch;
  // mid: [NH*32 n][128 k], GN = 2 (one group per head); dst blocks (h, sc) of [HC/4 kq][32 n] words
  prep_mat(P.w + R_MID, NH * 32, 128, 2, s8 + OF::M, (float*)(P.scratch + OF::M + 128 * C), [&](int n, int k) {
    return (((n >> 5) * NS + k / HC) * (HC / 4) + (k % HC) / 4) * 128 + (n & 31) * 4 + (k & 3); });
  // contract / out_proj: [C n][C k], GN = C/16; dst blocks per head (k / 32) of [8 kq][C n] words
  auto dk = [&](int n, int k) { return ((k >> 5) * 8 + ((k & 31) >> 2)) * C * 4 + n * 4 + (k & 3); };
  prep_mat(P.w + R_CON, C, C, C / 16, s8 + OF::K, (float*)(P.scratch + OF::K + C * C), dk);
  prep_mat(P.w + R_OUT, C, C, C / 16, s8 + OF::O, (float*)(P.scratch + OF::O + C * C), dk);
}
// <name>_i8 = mask 1, <name>_i8_<mask> for the others (nr_runtime picks by int8_swin)
#define I8K(C, S, ATTR, NAME)                                                                                            \
  extern "C" __global__ __launch_bounds__(256) ATTR void NAME##_i8(VarParams P) { swin_block<C, S, 1>(P); }              \
  extern "C" __global__ __launch_bounds__(256) ATTR void NAME##_i8_3(VarParams P) { swin_block<C, S, 3>(P); }            \
  extern "C" __global__ __launch_bounds__(256) ATTR void NAME##_i8_5(VarParams P) { swin_block<C, S, 5>(P); }            \
  extern "C" __global__ __launch_bounds__(256) ATTR void NAME##_i8_7(VarParams P) { swin_block<C, S, 7>(P); }            \
  extern "C" __global__ __launch_bounds__(256) ATTR void NAME##_i8_9(VarParams P) { swin_block<C, S, 9>(P); }            \
  extern "C" __global__ __launch_bounds__(256) ATTR void NAME##_i8_13(VarParams P) { swin_block<C, S, 13>(P); }          \
  extern "C" __global__ __launch_bounds__(256) ATTR void NAME##_i8_15(VarParams P) { swin_block<C, S, 15>(P); }          \
  extern "C" __global__ __launch_bounds__(256) void NAME##_i8_prep(VarParams P) { i8_prep<C>(P); i8_prep_qkv<C>(P); i8_prep_rest<C>(P); }
#ifdef BIGWPE
#define NOATTR __attribute__((amdgpu_waves_per_eu(BIGWPE)))
#else
#define NOATTR
#endif
I8K(256, false, NOATTR, k_swin_c256)
I8K(128, false, NOATTR, k_swin_c128)
I8K(256, true, NOATTR, k_swin_c256s)
I8K(128, true, NOATTR, k_swin_c128s)
I8K(64, false, __attribute__((amdgpu_waves_per_eu(8))), k_swin_c64)

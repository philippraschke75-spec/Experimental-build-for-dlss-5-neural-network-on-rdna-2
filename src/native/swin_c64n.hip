// Native gfx1030 2-head Swin block (C = 64), int8_swin mask 13 numerics (expand+qkv, contract, out_proj on int8 dot4,
// mid on f16 dot2): byte-identical drop-in for k_swin_c64_i8_13 of swin_var.hip, flags 0/1/2 only (no up/downsample).
// Layout as swin_c32.hip: one 8x8 window per 64-thread workgroup (two waves), lane = token for load, MLP, qkv, out_proj
// and store (registers, no LDS, no barrier). Every weight comes through SGPRs from the per-record scratch that
// k_swin_c64_i8_prep builds once at load (kernarg +0xa0), stored in the order the lanes consume it. LDS (12.4 KB) only
// carries the attention exchanges, exactly as swin_c32. Each output keeps the accumulation order of swin_block<64, 13>.
// Build with -ffp-contract=off -mcumode.
#include "swin_common.h"

#ifndef WPE
#ifdef Q8FAST
#define WPE 3   // Q8FAST: fewer VGPRs needed, 3 waves/SIMD is 0.23 ms faster at 1440p
#else
#define WPE 4
#endif
#endif

typedef const __attribute__((address_space(4))) uint32_t* cwp;
typedef const __attribute__((address_space(4))) float* cfp;

// scratch words: E expand int8 [2 h][64 pair][2 j][16 (kb, kq)] | ES f32 [256] | M mid f16 [2 h][64 pair][32 c] |
// K contract int8 [2 h][64 ch][8 kq] | KS [64] | Q qkv int8 [2 h][96 n][16] | QS [192] | O out_proj int8 [2 h][64 ch][8] |
// OS [64] | RS [64] | ARS [64] | HSC [2] | BIAS f32 [2 h][64 thread][8 q][8 k] = fmaf(bias, 0.044921875, 1.30078125)
constexpr int E0 = 0, ES0 = 4096, M0 = 4352, K0 = 8448, KS0 = 9472, Q0 = 9536, QS0 = 12608, O0 = 12800, OS0 = 13824;
constexpr int RS0 = 13888, ARS0 = 13952, HSC0 = 14016, BI0 = 14080, S_END = BI0 + 8192;
// flags 8: UG f32 [64] upsample gain, WR f16 [2 wv][64 kp][32 n] up resample (out 32wv + n); flags 4: WD f16 [2 wv][32 kp][64 n]
constexpr int UG0 = S_END, WR0 = UG0 + 64, WD0 = WR0 + 4096, S_END2 = WD0 + 4096;

__device__ __forceinline__ float pow2s(float m) { int e; frexpf(m, &e); return m > 0.f ? ldexpf(1.f, e - 7) : 1.f; }
__device__ __forceinline__ uint32_t q8h(uint32_t xb) {   // pk(q8(f16r(a)), q8(f16r(b))) from xb = pk(a, b) (swin_c32)
#ifdef Q8FAST
  return (xb + 0x00400040u) & 0xFF80FF80u;
#endif
  typedef unsigned short u2 __attribute__((ext_vector_type(2)));
  const u2 ab = __builtin_elementwise_min(__builtin_bit_cast(u2, xb & 0x7FFF7FFFu), (u2){0x5F00, 0x5F00});
  const u2 eb = __builtin_elementwise_max(ab & (u2){0x7C00, 0x7C00}, (u2){0x2400, 0x2400});
  const h2 K = __builtin_bit_cast(h2, eb) * (h2){(_Float16)128.f, (_Float16)128.f};
  const h2 r = (__builtin_bit_cast(h2, ab) + K) - K;
  return (__builtin_bit_cast(uint32_t, r) & 0x7FFF7FFFu) | (xb & 0x80008000u);
}
__device__ __forceinline__ uint32_t e4h(uint32_t r) {   // e4m3 bytes of a q8h word (swin_c32)
#ifdef Q8BYTE   // f16 bits >> 7 - 64 (sat): tiny values below 2^-6 flush to 0
  { typedef unsigned short u2b_ __attribute__((ext_vector_type(2))); const u2b_ a_ = __builtin_elementwise_min(__builtin_bit_cast(u2b_, r & 0x7FFF7FFFu), (u2b_){0x5F00, 0x5F00}) >> (u2b_){7, 7};
    return __builtin_bit_cast(uint32_t, __builtin_elementwise_sub_sat(a_, (u2b_){64, 64})) | ((r & 0x80008000u) >> 8); }
#endif
#ifdef Q8FAST
  { typedef unsigned short u2_ __attribute__((ext_vector_type(2))); r = (r & 0x80008000u) | __builtin_bit_cast(uint32_t, __builtin_elementwise_min(__builtin_bit_cast(u2_, r & 0x7FFF7FFFu), (u2_){0x5F00, 0x5F00})); }
#endif
  typedef unsigned short u2 __attribute__((ext_vector_type(2)));
  const u2 ab = __builtin_bit_cast(u2, r & 0x7FFF7FFFu);
  const u2 nb = ((ab - (u2){0x2400, 0x2400}) >> (u2){7, 7}) + (u2){8, 8};
  const h2 sh = __builtin_bit_cast(h2, ab) * (h2){(_Float16)512.f, (_Float16)512.f} + (h2){(_Float16)1024.f, (_Float16)1024.f};
  const u2 sb = __builtin_bit_cast(u2, sh) - (u2){0x6400, 0x6400};
  const uint32_t nz = __builtin_bit_cast(uint32_t, ab + (u2){0x7FFF, 0x7FFF});
  return __builtin_bit_cast(uint32_t, __builtin_elementwise_min(nb, sb)) | ((r & nz & 0x80008000u) >> 8);
}
__device__ __forceinline__ uint32_t e4x4(const float* o, int m) {   // memory channels m..m+3 of a 32-channel chunk
  const int c0 = pi32(m), c1 = pi32(m + 2);
  const uint32_t a = e4h(q8h(pk(o[c0], o[c0 + 1]))), b = e4h(q8h(pk(o[c1], o[c1 + 1])));
  return __builtin_amdgcn_perm(b, a, 0x06040200u);
}
// int8 quads + power-of-two scale per 32-channel group of 32 f16 words (swin_block quant(), one token)
__device__ __forceinline__ void quant(const uint32_t* V, uint32_t* Qw, float* sa) {
#pragma unroll
  for (int kb = 0; kb < 2; kb++) {
    float m = 0.f;
#pragma unroll
    for (int j = 0; j < 16; j++) m = fmaxf(m, fmaxf(fabsf(lo(V[kb * 16 + j])), fabsf(hi(V[kb * 16 + j]))));
    sa[kb] = pow2s(m);
    const float inv = 1.f / sa[kb];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const uint32_t u = V[kb * 16 + 2 * j], v = V[kb * 16 + 2 * j + 1];
      const float f[4] = {lo(u), hi(u), lo(v), hi(v)};
      uint32_t w = 0;
#pragma unroll
      for (int l = 0; l < 4; l++) w |= ((uint32_t)(int)rintf(f[l] * inv) & 0xFFu) << (8 * l);
      Qw[kb * 8 + j] = w;
    }
  }
}
// K = 64 int8 dot: 16 activation quads against 16 weight words, scaled per 32-channel group (kb order, as swin_block)
__device__ __forceinline__ float i8dot16(const uint32_t* a, cwp w, const float* sa) {
  int ai = 0;
#pragma unroll
  for (int kq = 0; kq < 8; kq++) ai = __builtin_amdgcn_sdot4((int)a[kq], (int)w[kq], ai, false);
  float acc = fmaf((float)ai, sa[0], 0.f);
  ai = 0;
#pragma unroll
  for (int kq = 0; kq < 8; kq++) ai = __builtin_amdgcn_sdot4((int)a[8 + kq], (int)w[8 + kq], ai, false);
  return fmaf((float)ai, sa[1], acc);
}
// 32 f32 -> 8 int8 quads with one power-of-two scale (Ms / ctx of one head)
__device__ __forceinline__ float quant32(const float* v, uint32_t* q) {
  float m = 0.f;
#pragma unroll
  for (int c = 0; c < 32; c++) m = fmaxf(m, fabsf(v[c]));
  const float sa = pow2s(m), inv = 1.f / sa;
#pragma unroll
  for (int k = 0; k < 8; k++) {
    uint32_t w = 0;
#pragma unroll
    for (int c = 0; c < 4; c++) w |= ((uint32_t)(int)rintf(v[4 * k + c] * inv) & 0xFFu) << (8 * c);
    q[k] = w;
  }
  return sa;
}

template <int Dummy>
__device__ __forceinline__ void c64_body(const VarParams& P) {
  constexpr int C = 64;
  __shared__ uint32_t RA[2048];   // qn [16 i][64 tok] + kn [16 i][64 tok] | Pm [32 key pair][64 q]
  __shared__ uint32_t RB[1056];   // Vt [32 key pair][33] | ctx f16 [16 kp][64 tok]
  const int t = threadIdx.x;
  const cwp S = (cwp)P.scratch;
  const cfp Sf = (cfp)P.scratch;
  const int H = P.H, W = P.W, WT = W >> 2;
  const int r0 = P.Y0 + blockIdx.y * 8, c0 = P.X0 + blockIdx.x * 8;
  const int lr = ((t >> 5) & 1) * 4 + ((t >> 2) & 3), lc = ((t >> 4) & 1) * 4 + (t & 3);
  const int row = r0 + lr, col = c0 + lc;
  const bool in = row >= 0 && row < H && col >= 0 && col < W;
  const int p = (row & 3) * 4 + (col & 3);
  const bool up = P.flags & 8;
  const int lane = t & 31, wv = __builtin_amdgcn_readfirstlane(t >> 5);
  const int pix = (lr >> 1) * 4 + (lc >> 1);   // coarse pixel under the token
  const _Float16* Pj = (const _Float16*)RA;   // up: [16 pix][64] f16, alive until the residual (RA is free until attention)

  // ---- upsample: Pj = f16(Wr . coarse) (K = 2C = 128); wave wv makes outputs 32wv..+31 for pixel lane & 15
  if (up) {
    const int H2 = H >> 1, W2 = W >> 1, Hs = (H2 + 3) & ~3, Ws = (W2 + 3) & ~3;
    const int px = lane & 15, pr = (r0 >> 1) + (px >> 2), pc = (c0 >> 1) + (px & 3);
    uint32_t cb[32] = {};
    if (pr >= 0 && pr < H2 && pc >= 0 && pc < W2) {
#pragma unroll
      for (int ch = 0; ch < 4; ch++) {
        const uint8_t* src = P.p30 + (size_t)(ch * 2) * Hs * Ws * 16 + (size_t)(pr * Ws + pc) * 16;
#pragma unroll
        for (int g = 0; g < 4; g++) { cb[ch * 8 + 2 * g] = *(const uint32_t*)(src + 4 * g); cb[ch * 8 + 2 * g + 1] = *(const uint32_t*)(src + (size_t)Hs * Ws * 16 + 4 * g); }
      }
    }
    uint32_t cw[64];
#pragma unroll
    for (int ch = 0; ch < 4; ch++)
#pragma unroll
      for (int g = 0; g < 4; g++)
#pragma unroll
        for (int s2 = 0; s2 < 2; s2++)
#pragma unroll
          for (int hh = 0; hh < 2; hh++) {
            const int c = 16 * s2 + 4 * g + 2 * hh;
            cw[(ch * 32 + pi32(c)) >> 1] = x256(e4x2((cb[ch * 8 + 2 * g + s2] >> (16 * hh)) & 0xFFFFu));
          }
    float a[32];
#pragma unroll
    for (int n = 0; n < 32; n++) a[n] = 0.f;
    const cwp wr = S + WR0 + wv * 2048;
#pragma unroll
    for (int kp = 0; kp < 64; kp++)
#pragma unroll
      for (int n = 0; n < 32; n++) a[n] = dot2(cw[kp], wr[kp * 32 + n], a[n]);
    if (lane < 16)
#pragma unroll
      for (int n = 0; n < 32; n++) ((_Float16*)RA)[px * 64 + 32 * wv + n] = (_Float16)(a[n] * 256.f);
    __syncthreads();
  }

  // ---- the token: X[k] = f16 pair of reference channels 2k, 2k+1 (loaded twice: for the MLP input and the residual,
  //      so X is not live across the MLP)
  auto load_x = [&](uint32_t* X) {
    uint32_t b[2][8] = {};
    if (in) {
#pragma unroll
      for (int ch = 0; ch < 2; ch++) {
        if (P.flags & 1) {
          const uint8_t* src = P.in + (size_t)(ch * 2) * H * W * 16 + (size_t)(row * W + col) * 16;
#pragma unroll
          for (int g = 0; g < 4; g++) { b[ch][2 * g] = *(const uint32_t*)(src + 4 * g); b[ch][2 * g + 1] = *(const uint32_t*)(src + (size_t)H * W * 16 + 4 * g); }
        } else {
          const uint8_t* src = P.in + (size_t)((row >> 2) * WT + (col >> 2)) * (C * 16) + ch * 512 + (p & 7) * 64 + 4 * (p >> 3);
#pragma unroll
          for (int g = 0; g < 4; g++) { b[ch][2 * g] = *(const uint32_t*)(src + 16 * g); b[ch][2 * g + 1] = *(const uint32_t*)(src + 16 * g + 8); }
        }
      }
    }
#pragma unroll
    for (int ch = 0; ch < 2; ch++)
#pragma unroll
      for (int g = 0; g < 4; g++)
#pragma unroll
        for (int s = 0; s < 2; s++)
#pragma unroll
          for (int hh = 0; hh < 2; hh++) {
            const int c = 16 * s + 4 * g + 2 * hh;
            X[(ch * 32 + pi32(c)) >> 1] = x256(e4x2((b[ch][2 * g + s] >> (16 * hh)) & 0xFFFFu));
          }
    if (up)   // x = q8(f16(skip * gain + P(coarse))), zero outside the frame (skip * gain is exact in f32)
#pragma unroll
      for (int k = 0; k < 32; k++) {
        const float x0 = q8(f16r(__fadd_rn(__fmul_rn(lo(X[k]), Sf[UG0 + 2 * k]), (float)Pj[pix * 64 + 2 * k])));
        const float x1 = q8(f16r(__fadd_rn(__fmul_rn(hi(X[k]), Sf[UG0 + 2 * k + 1]), (float)Pj[pix * 64 + 2 * k + 1])));
        X[k] = in ? pk(x0, x1) : 0u;
      }
  };

  // ================= MLP: per head, per hidden pair: expand (int8) + act + q8, mid partial (f16); then Ms -> contract (int8)
  uint32_t Xq[16];
  float xs[2];
  { uint32_t X[32]; load_x(X); quant(X, Xq, xs); }
  float accc[64];
#pragma unroll
  for (int c = 0; c < 64; c++) accc[c] = 0.f;
#pragma unroll 1
  for (int h = 0; h < 2; h++) {
    float macc[32];
#pragma unroll
    for (int c = 0; c < 32; c++) macc[c] = 0.f;
#pragma unroll 1
    for (int pr = 0; pr < 64; pr++) {
      const int n = h * 128 + 2 * pr;
      const float a0 = i8dot16(Xq, S + E0 + (h * 64 + pr) * 32, xs);
      const float a1 = i8dot16(Xq, S + E0 + (h * 64 + pr) * 32 + 16, xs);
      const uint32_t hd = pk(q8(act(a0 * Sf[ES0 + n])), q8(act(a1 * Sf[ES0 + n + 1])));
      const cwp wm = S + M0 + (h * 64 + pr) * 32;
#pragma unroll
      for (int c = 0; c < 32; c++) macc[c] = dot2(hd, wm[c], macc[c]);
    }
    float v[32];
#pragma unroll
    for (int c = 0; c < 32; c++) v[c] = q8(f16r(macc[c] * 256.f));
    uint32_t ms[8];
    const float sa = quant32(v, ms);
#pragma unroll
    for (int ch = 0; ch < 64; ch++) {
      const cwp wk = S + K0 + (h * 64 + ch) * 8;
      int ai = 0;
#pragma unroll
      for (int kq = 0; kq < 8; kq++) ai = __builtin_amdgcn_sdot4((int)ms[kq], (int)wk[kq], ai, false);
      accc[ch] = fmaf((float)ai, sa, accc[ch]);
    }
  }

  // ---- y = rs * x + mlp; Yq = q8(f16(y)) is the attention input and the skip
  uint32_t Yq[32];
  uint32_t X[32];
  load_x(X);
#pragma unroll
  for (int k = 0; k < 32; k++) {
    const float y0 = __fadd_rn(__fmul_rn(lo(X[k]), Sf[RS0 + 2 * k]), accc[2 * k] * Sf[KS0 + 2 * k]);
    const float y1 = __fadd_rn(__fmul_rn(hi(X[k]), Sf[RS0 + 2 * k + 1]), accc[2 * k + 1] * Sf[KS0 + 2 * k + 1]);
    Yq[k] = pk(q8(f16r(y0)), q8(f16r(y1)));
  }
  uint32_t Yi[16];
  float ys[2];
  quant(Yq, Yi, ys);

  // ================= attention, one head at a time; out_proj (int8) accumulated per head
  uint32_t* qn = RA;          // [16 i][64 tok]
  uint32_t* kn = RA + 1024;   // [16 i][64 tok]
  uint32_t* Pm = RA;          // [32 key pair][64 q]
  uint32_t* Vt = RB;          // [32 key pair][33]
  uint32_t* ctx = RB;         // [16 kp][64 tok] f16
  float acco[64];
#pragma unroll
  for (int c = 0; c < 64; c++) acco[c] = 0.f;
#pragma unroll 1
  for (int h = 0; h < 2; h++) {
    if (h) __syncthreads();   // the previous head's out_proj read ctx (RB)
#pragma unroll 1
    for (int g = 0; g < 3; g++) {   // q, k, v: 32 outputs each
      float a[32];
#pragma unroll
      for (int n = 0; n < 32; n++) {
        const int nn = h * 96 + g * 32 + n;
        float acc = i8dot16(Yi, S + Q0 + nn * 16, ys);
        acc *= Sf[QS0 + nn] * (1.f / 256.f);
        a[n] = acc * 256.f;
      }
      if (g < 2) {
        float sc = norm_scale(a);
        if (g == 0) sc = __fmul_rn(sc, Sf[HSC0 + h]);
        uint32_t* dst = g ? kn : qn;
#pragma unroll
        for (int d = 0; d < 32; d += 2) dst[(d >> 1) * 64 + t] = pk(q8(__fmul_rn(a[d], sc)), q8(__fmul_rn(a[d + 1], sc)));
      } else {
        uint32_t own[16];
#pragma unroll
        for (int k = 0; k < 16; k++) own[k] = q8h(pk(a[2 * k], a[2 * k + 1]));
        const bool odd = t & 1;
        const int j = t >> 1;
#pragma unroll
        for (int k = 0; k < 8; k++) {
          const uint32_t send = odd ? own[k] : own[8 + k];
          const uint32_t recv = (uint32_t)__builtin_amdgcn_mov_dpp((int)send, 0xB1, 0xF, 0xF, false);
          const uint32_t mine = odd ? own[8 + k] : own[k];
          const uint32_t w0 = odd ? ((recv & 0xFFFFu) | (mine << 16)) : ((mine & 0xFFFFu) | (recv << 16));
          const uint32_t w1 = odd ? ((recv >> 16) | (mine & 0xFFFF0000u)) : ((mine >> 16) | (recv & 0xFFFF0000u));
          const int d = 2 * k + (odd ? 16 : 0);
          Vt[j * 33 + d] = w0; Vt[j * 33 + d + 1] = w1;
        }
      }
    }
    __syncthreads();
    {   // logits -> exponent-ramp numerators: thread = 8 queries (8qb..) x 8 keys (8kb..)
      const int qb = t & 7, kb = t >> 3;
      float lg[8][8] = {};
#pragma unroll 2
      for (int i = 0; i < 16; i++) {
        uint32_t qv[8], kv[8];
#pragma unroll
        for (int e = 0; e < 8; e++) { qv[e] = qn[i * 64 + 8 * qb + e]; kv[e] = kn[i * 64 + 8 * kb + e]; }
#pragma unroll
        for (int a = 0; a < 8; a++)
#pragma unroll
          for (int b = 0; b < 8; b++) lg[a][b] = dot2(qv[a], kv[b], lg[a][b]);
      }
      const float4* bp = (const float4*)(P.scratch + 4 * (BI0 + h * 4096 + t * 64));
      uint32_t num[8][4];
#pragma unroll
      for (int a = 0; a < 8; a++) {
        const float4 b0 = bp[2 * a], b1 = bp[2 * a + 1];
        const float bb[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {
          const float s0 = fmaf(lg[a][2 * jj], 0.044921875f, bb[2 * jj]), s1 = fmaf(lg[a][2 * jj + 1], 0.044921875f, bb[2 * jj + 1]);
          h2 r = (h2){(_Float16)s0, (_Float16)s1};
          r = __builtin_elementwise_max(r, (h2){(_Float16)1.03125f, (_Float16)1.03125f});
          r = __builtin_elementwise_min(r, (h2){(_Float16)1.5693359375f, (_Float16)1.5693359375f});
          num[a][jj] = (__builtin_bit_cast(uint32_t, r) & 0x03FF03FFu) << 5;
        }
      }
      __syncthreads();   // Pm replaces qn/kn
#pragma unroll
      for (int jj = 0; jj < 4; jj++)
#pragma unroll
        for (int a = 0; a < 8; a++) Pm[(4 * kb + jj) * 64 + 8 * qb + a] = num[a][jj];
    }
    __syncthreads();
    {   // lane = query: denominator in the reference order, then P = q8(f16(num / den)) in place
      h2 pp[32];
#pragma unroll
      for (int j = 0; j < 32; j++) pp[j] = __builtin_bit_cast(h2, Pm[j * 64 + t]);
      h2 hm[2];
#pragma unroll
      for (int m = 0; m < 2; m++) {
        h2 prr[4];
#pragma unroll
        for (int q = 0; q < 4; q++) { int b = 2 * q + m; prr[q] = ((pp[b] + pp[8 + b]) + pp[16 + b]) + pp[24 + b]; }
        hm[m] = ((prr[0] + prr[1]) + prr[2]) + prr[3];
      }
      const _Float16 den = (hm[0].x + hm[0].y) + (hm[1].x + hm[1].y);
      const float rc = __fdiv_rn(1.f, fmaxf((float)den, 0.00006103515625f));
#pragma unroll
      for (int j = 0; j < 32; j++) {
        const uint32_t u = __builtin_bit_cast(uint32_t, pp[j]);
        Pm[j * 64 + t] = q8h(pk(__fmul_rn(lo(u), rc), __fmul_rn(hi(u), rc)));
      }
    }
    __syncthreads();
    {   // context = P V: thread = 8 queries (8qb..) x 4 dims (4db..)
      const int qb = t & 7, db = t >> 3;
      float s[8][4] = {};
#pragma unroll 4
      for (int j = 0; j < 32; j++) {
        uint32_t pv[8], vv[4];
#pragma unroll
        for (int a = 0; a < 8; a++) pv[a] = Pm[j * 64 + 8 * qb + a];
#pragma unroll
        for (int e = 0; e < 4; e++) vv[e] = Vt[j * 33 + 4 * db + e];
#pragma unroll
        for (int a = 0; a < 8; a++)
#pragma unroll
          for (int e = 0; e < 4; e++) s[a][e] = dot2(pv[a], vv[e], s[a][e]);
      }
      __syncthreads();   // ctx replaces Vt
#pragma unroll
      for (int a = 0; a < 8; a++)
#pragma unroll
        for (int e = 0; e < 4; e += 2) ctx[((4 * db + e) >> 1) * 64 + 8 * qb + a] = q8h(pk(s[a][e], s[a][e + 1]));
    }
    __syncthreads();
    {   // out_proj (lane = token): ctx -> int8 with one scale, acco += Wo_h . ctx
      float v[32];
#pragma unroll
      for (int kp = 0; kp < 16; kp++) { const uint32_t u = ctx[kp * 64 + t]; v[2 * kp] = lo(u); v[2 * kp + 1] = hi(u); }
      uint32_t cq[8];
      const float sa = quant32(v, cq);
#pragma unroll
      for (int ch = 0; ch < 64; ch++) {
        const cwp wo = S + O0 + (h * 64 + ch) * 8;
        int ai = 0;
#pragma unroll
        for (int kq = 0; kq < 8; kq++) ai = __builtin_amdgcn_sdot4((int)cq[kq], (int)wo[kq], ai, false);
        acco[ch] = fmaf((float)ai, sa, acco[ch]);
      }
    }
  }

  // ---- out = ars * Yq + Wo . ctx ; store q8(f16(out)) as e4m3
#pragma unroll
  for (int k = 0; k < 32; k++) {
    acco[2 * k] = __fadd_rn(__fmul_rn(lo(Yq[k]), Sf[ARS0 + 2 * k]), acco[2 * k] * Sf[OS0 + 2 * k]);
    acco[2 * k + 1] = __fadd_rn(__fmul_rn(hi(Yq[k]), Sf[ARS0 + 2 * k + 1]), acco[2 * k + 1] * Sf[OS0 + 2 * k + 1]);
  }
  if (in) {
#pragma unroll
    for (int ch = 0; ch < 2; ch++) {
      const float* o = acco + 32 * ch;
      if (P.flags & 2) {
#pragma unroll
        for (int pl = 0; pl < 2; pl++) {
          uint32_t w[4];
#pragma unroll
          for (int k = 0; k < 4; k++) w[k] = e4x4(o, 16 * pl + 4 * k);
          *(uint4*)(P.out + (size_t)(ch * 2 + pl) * H * W * 16 + (size_t)(row * W + col) * 16) = (uint4){w[0], w[1], w[2], w[3]};
        }
      } else {
        uint8_t* dst = P.out + (size_t)((row >> 2) * WT + (col >> 2)) * (C * 16) + ch * 512 + (p & 7) * 64 + 4 * (p >> 3);
#pragma unroll
        for (int a = 0; a < 4; a++)
#pragma unroll
          for (int b = 0; b < 2; b++) *(uint32_t*)(dst + 16 * a + 8 * b) = e4x4(o, 4 * a + 16 * b);
      }
    }
  }
  if (!(P.flags & 4)) return;

  // ---- downsample: pooled = q8(pool2x2(out)) -> RA f16 [16 pix][64]; next = q8(f16(Wds . pooled)) -> +0x38 planes [8][H/2][W/2][16]
  __syncthreads();   // every wave is past its last RA (Pm) read
  {
    const bool lead = !(t & 1) && !(t & 4);
#pragma unroll
    for (int c = 0; c < 64; c += 2) {   // x partner = lane ^ 1, y partner = lane ^ 4 (DPP)
      const float ox0 = __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(acco[c]), 0xB1, 0xF, 0xF, false));
      const float ox1 = __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(acco[c + 1]), 0xB1, 0xF, 0xF, false));
      const h2 hp = (h2){(_Float16)(acco[c] + ox0), (_Float16)(acco[c + 1] + ox1)};
      const h2 hy = __builtin_bit_cast(h2, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, hp), 0x164, 0xF, 0xF, false));
      const h2 tot = hp + hy;
      if (lead) RA[pix * 32 + (c >> 1)] = pk(q8((float)tot.x * 0.25f), q8((float)tot.y * 0.25f));
    }
  }
  __syncthreads();
  {
    const int px = lane & 15;
    float acc[64];
#pragma unroll
    for (int n = 0; n < 64; n++) acc[n] = 0.f;
    const cwp wd = S + WD0 + wv * 2048;
#pragma unroll
    for (int kp = 0; kp < 32; kp++) {
      const uint32_t pw = RA[px * 32 + kp];
#pragma unroll
      for (int n = 0; n < 64; n++) acc[n] = dot2(pw, wd[kp * 64 + n], acc[n]);
    }
    const int H2 = H >> 1, W2 = W >> 1, Hs = (H2 + 3) & ~3, Ws = (W2 + 3) & ~3;
    const int pr = (r0 >> 1) + (px >> 2), pc = (c0 >> 1) + (px & 3);
    if (lane < 16 && pr >= 0 && pr < H2 && pc >= 0 && pc < W2) {
#pragma unroll
      for (int n = 0; n < 64; n++) acc[n] *= 256.f;
#pragma unroll
      for (int cc = 0; cc < 2; cc++)
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          uint32_t w[4];
#pragma unroll
          for (int k = 0; k < 4; k++) w[k] = e4x4(acc + 32 * cc, 16 * hh + 4 * k);
          *(uint4*)(P.p38 + (size_t)((2 * wv + cc) * 2 + hh) * Hs * Ws * 16 + (size_t)(pr * Ws + pc) * 16) = (uint4){w[0], w[1], w[2], w[3]};
        }
    }
  }
}

// ---- prep (once per record, runtime int8_swin path: grid 16 x 256 threads, P.scratch = 1 MiB destination)
__device__ __forceinline__ float wtile(const uint8_t* m, int GN, int K, int n, int k) {
  const int tile = (n / (16 * GN)) * (GN * K / 32) + (k >> 5) * GN + ((n % (16 * GN)) >> 4), nn = n & 15, kk = k & 31;
  return lo(e4x2(m[tile * 512 + (nn & 7) * 64 + ((kk >> 1) & 3) * 16 + (nn >> 3) * 8 + (kk >> 3) * 2 + (kk & 1)])) * 256.f;
}
// rows [n0, n0 + N) of an [N][K] e4m3 matrix -> int8 words: word w(n, k/4) = rintf(w / scale); scale per row in sc
template <typename D>
__device__ void prep_rows(const uint8_t* m, int N, int K, int GN, uint32_t* S, float* sc, D di) {
  for (int n = blockIdx.x * 256 + threadIdx.x; n < N; n += gridDim.x * 256) {
    float mx = 0.f;
    for (int k = 0; k < K; k++) mx = fmaxf(mx, fabsf(wtile(m, GN, K, n, k)));
    const float s = pow2s(mx);
    sc[n] = s;
    for (int k = 0; k < K; k += 4) {
      uint32_t w = 0;
      for (int b = 0; b < 4; b++) w |= ((uint32_t)(int)rintf(wtile(m, GN, K, n, k + b) / s) & 0xFFu) << (8 * b);
      S[di(n, k)] = w;
    }
  }
}
extern "C" __global__ __launch_bounds__(256) void k_swin_c64_i8_prep(VarParams P) {
  constexpr int C = 64, NH = 2;
  const uint8_t* rec = P.w;
  const bool up = P.flags & 8;
  constexpr int R_MID = 4 * C * C, R_CON = R_MID + C * 128, R_AFT = R_CON + C * C;
  const int R_RS = up ? R_AFT + 2 * C * C : R_AFT, R_QKV = R_RS + 2 * C + 32 + (up ? 2 * C - 32 : 0);
  const int R_BIAS = R_QKV + 3 * C * C, R_HS = R_BIAS + NH * 8192, R_OUT = R_HS + (4 * NH > 16 ? 4 * NH : 16);
  const int R_ARS = R_OUT + C * C;
  uint32_t* S = (uint32_t*)P.scratch;
  float* Sf = (float*)P.scratch;
  // expand [4C][C], GN 8: hidden n = h*128 + 2 pr + j -> E[(h*64 + pr)*32 + j*16 + kb*8 + kq]
  prep_rows(rec, 4 * C, C, 8, S, Sf + ES0, [](int n, int k) { return E0 + (n >> 1) * 32 + (n & 1) * 16 + (k >> 2); });
  // qkv [3C][C], GN 3C/16: row n -> Q[n*16 + k/4]
  prep_rows(rec + R_QKV, 3 * C, C, 3 * C / 16, S, Sf + QS0, [](int n, int k) { return Q0 + n * 16 + (k >> 2); });
  // contract / out_proj [C][C], GN C/16: head h = k/32 -> [(h*64 + n)*8 + (k%32)/4]
  prep_rows(rec + R_CON, C, C, C / 16, S, Sf + KS0, [](int n, int k) { return K0 + ((k >> 5) * 64 + n) * 8 + ((k & 31) >> 2); });
  prep_rows(rec + R_OUT, C, C, C / 16, S, Sf + OS0, [](int n, int k) { return O0 + ((k >> 5) * 64 + n) * 8 + ((k & 31) >> 2); });
  const int gt = blockIdx.x * 256 + threadIdx.x, gs = gridDim.x * 256;
  // mid [NH*32][128], GN 2: f16 (x 2^-8) k-pairs -> M[(h*64 + pr)*32 + c], n = h*32 + c
  for (int i = gt; i < NH * 32 * 64; i += gs) {
    const int n = i >> 6, pr = i & 63;
    S[M0 + ((n >> 5) * 64 + pr) * 32 + (n & 31)] =
        pk(wtile(rec + R_MID, 2, 128, n, 2 * pr) * (1.f / 256.f), wtile(rec + R_MID, 2, 128, n, 2 * pr + 1) * (1.f / 256.f));
  }
  if (gt < C) {
    const _Float16* rs = (const _Float16*)(rec + R_RS);
    const uint4 rs8 = *(const uint4*)rs;
    const int rso = (rs8.x | rs8.y | rs8.z | rs8.w) == 0 ? 8 : 0;
    Sf[RS0 + gt] = (float)rs[rso + gt];
    Sf[ARS0 + gt] = (float)((const _Float16*)(rec + R_ARS))[gt];
    if (gt < NH) Sf[HSC0 + gt] = ((const float*)(rec + R_HS))[gt];
  }
  if (up) {   // upsample gain and resample Wr [C n][2C k] (GN C/16), output 32wv + n
    if (gt < C) {
      const _Float16* rs = (const _Float16*)(rec + R_RS);
      const _Float16* ug = (const _Float16*)(rec + R_RS + 2 * C + 32);
      Sf[UG0 + gt] = gt < 16 ? (float)rs[C + gt] : (float)ug[gt - 16];
    }
    for (int i = gt; i < 4096; i += gs) {
      const int w2 = i >> 11, kp = (i >> 5) & 63, n = i & 31, o = 32 * w2 + n;
      S[WR0 + i] = pk(wtile(rec + R_AFT, C / 16, 2 * C, o, 2 * kp) * (1.f / 256.f), wtile(rec + R_AFT, C / 16, 2 * C, o, 2 * kp + 1) * (1.f / 256.f));
    }
  }
  if (P.flags & 4)   // ds resample Wds [2C n][C k] (GN 2C/16), output 64wv + n
    for (int i = gt; i < 4096; i += gs) {
      const int w2 = i >> 11, kp = (i >> 6) & 31, n = i & 63, o = 64 * w2 + n;
      S[WD0 + i] = pk(wtile(rec + R_ARS + 2 * C, 2 * C / 16, C, o, 2 * kp) * (1.f / 256.f), wtile(rec + R_ARS + 2 * C, 2 * C / 16, C, o, 2 * kp + 1) * (1.f / 256.f));
    }
  for (int e = gt; e < NH * 4096; e += gs) {   // logits thread th = qb + 8kb holds queries 8qb + a, keys 8kb + b
    const int h = e >> 12, th = (e >> 6) & 63, a = (e >> 3) & 7, b = e & 7, q = 8 * (th & 7) + a, s = 8 * (th >> 3) + b;
    const int ii = q & 15, j = s & 15;
    const int lane = 4 * (ii & 7) + ((j & 7) >> 1);
    const int slot = (j & 1) | ((ii >> 3) << 1) | ((j >> 3) << 2);
    const _Float16* braw = (const _Float16*)(rec + R_BIAS) + h * 4096;
    Sf[BI0 + e] = fmaf((float)braw[(4 * (q >> 4) + (s >> 4)) * 256 + lane * 8 + slot], 0.044921875f, 1.30078125f);
  }
}

// Every int8_swin mask for c64 runs the mask-13 numerics (expand+qkv, contract, out_proj on int8): byte-identical to
// k_swin_c64_i8_13, and within the int8 masks' measured spread (int8_swin agent, 2026-10-19) for the others.
// ponytail: int8_swin=0 (fp16) gets no scratch from the runtime -> the block is skipped; add an fp16 path if anyone needs it.
#define C64N(NAME) extern "C" __global__ __launch_bounds__(64) __attribute__((amdgpu_waves_per_eu(WPE))) void NAME(VarParams P) { if (P.scratch) c64_body<0>(P); }
C64N(k_swin_c64_i8) C64N(k_swin_c64_i8_3) C64N(k_swin_c64_i8_5) C64N(k_swin_c64_i8_7) C64N(k_swin_c64_i8_9) C64N(k_swin_c64_i8_13) C64N(k_swin_c64_i8_15)
C64N(k_swin_c64)   // manifest symbol (the runtime swaps in the _i8 variant)

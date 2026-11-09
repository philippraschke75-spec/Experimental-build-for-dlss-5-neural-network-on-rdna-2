// Native gfx1030 single-head Swin block (Swin1H, dlssnr/layers/swin.py) for k_swin_var<32>: drop-in for
// _Z10k_swin_varILi32ELb0EEv9VarParams (same VarParams kernarg, grid = one 8x8 window per workgroup, 64 threads).
// C = 32 specifics of the single-head block:
//  - no grouped mid projection: expand 32 -> 128 (+SiLU, q8), contract 128 -> 32;
//  - the attention skip is the unrounded f32 MLP output y (qkv reads q8(f16(y)));
//  - upsample (flags 8): blend = f16(skip * gain + P(coarse)) is the MLP residual, q8(blend) the matmul input;
//    gain = upsample_gain[0..31];
//  - record (extract.py _swin_layout, heads = 1): expand 4096 B | contract 4096 B | [resample 2048 B if up] |
//    residual_scale 96 B | [upsample_gain 64 B if up] | qkv 3072 B | bias 8192 B | scalars 16 B | out_proj 1024 B |
//    attn_residual_scale | [ds resample 2048 B].
// perf-c32b layout: one window per 64-thread workgroup (two waves), lane = token for the load, MLP, qkv, out_proj and
// store, so those run from registers with no LDS and no barrier. Every weight operand is wave-uniform and comes through
// SGPRs (s_load from the f16 shadow that k_swin_c32_prep builds once per record at load, kernarg +0xa0). LDS (12.4 KB,
// five workgroups per CU) only carries the attention exchanges: qn/kn -> logits (8 queries x 8 keys per thread) ->
// numerators -> denominator + normalise (lane = query) -> P V (8 queries x 4 dims per thread) -> ctx. Each output keeps
// the reference accumulation order of the previous 256-thread kernel (byte-identical results).
// Build with -ffp-contract=off -mcumode (build.sh).
#include "swin_common.h"

#ifndef WPE
#ifndef WPE
#define WPE 5
#endif
#endif
constexpr int HPG = 2;   // hidden pairs per MLP step

// f16 weight shadow (words = half2 k-pairs as put16 stages them), built by k_swin_c32_prep:
//  WE [128 n][16 kp]      expand, hidden n             WC [64 G][32 c]       contract, hidden pair G
//  WQ [3 g][16 kp][32 n]  qkv, output 32g + n           WO [16 kp][32 c]      out_proj
//  WR [2 h][32 kp][16 n]  up resample, out 16h + n      WD [2 h][16 kp][32 n] ds resample, out 32h + n
//  RSC/ARS/UG [32] f32 residual scale / attn residual scale / upsample gain, HSC f32 head scale,
//  BIAS [64 thread][8 q][8 k] f32 = fmaf(bias, 0.044921875, 1.30078125) in the logits thread order.
constexpr int S_WE = 0, S_WC = S_WE + 2048, S_WQ = S_WC + 2048, S_WO = S_WQ + 1536, S_WR = S_WO + 512, S_WD = S_WR + 1024;
constexpr int S_RSC = S_WD + 1024, S_ARS = S_RSC + 32, S_UG = S_ARS + 32, S_HSC = S_UG + 32, S_BIAS = S_HSC + 32;
constexpr int S_END = S_BIAS + 4096;
static_assert(S_BIAS % 4 == 0, "bias rows are read as float4");

typedef const __attribute__((address_space(4))) uint32_t* cwp;   // constant address space: wave-uniform -> s_load
typedef const __attribute__((address_space(4))) float* cfp;

// acc[n] = dot2(a[k], w[k * N + n], acc[n]) for k = 0..K-1 in order (the reference order per output). The weight row
// comes through SGPRs, one row ahead; the empty asm pins each row's dot2s before the scheduling barrier, so the compiler
// neither sinks the math nor hoists all K * N scalar loads (SGPR spills).
template <int K, int N>
__device__ __forceinline__ void gemv(float* acc, const uint32_t* a, cwp w) {
  uint32_t cur[N], nxt[N];
#pragma unroll
  for (int n = 0; n < N; n++) cur[n] = w[n];
  __builtin_amdgcn_sched_barrier(0);
#pragma unroll
  for (int k = 0; k < K; k++) {
    if (k + 1 < K)
#pragma unroll
      for (int n = 0; n < N; n++) nxt[n] = w[(k + 1) * N + n];
#pragma unroll
    for (int n = 0; n < N; n++) { acc[n] = dot2(a[k], cur[n], acc[n]); asm volatile("" : "+v"(acc[n])); }
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int n = 0; n < N; n++) cur[n] = nxt[n];
  }
}

// q8 for values known finite (no NaN guard), rounding by (a + K) - K with K = 2^(max(e, -6) + 20): one f32 rounding
// to 3 mantissa bits (2^-9 steps below 2^-6), ties to even = q8's bit trick; checked for every finite f32 (q8f_chk.py)
__device__ __forceinline__ float q8f(float x) {
#ifdef Q8FAST   // f16 + 3-bit mantissa only (2 int ops): keeps the look (CPU 45.3 dB to NVIDIA), no clamp (only swin needs it), no subnormal grid / ties-even
  { const uint32_t q_ = ((uint32_t)__builtin_bit_cast(unsigned short, (_Float16)x) + 0x40u) & 0xFF80u; return (float)__builtin_bit_cast(_Float16, (unsigned short)q_); }
#endif
  const float a = fminf(fabsf(x), 448.f);
  const float K = __uint_as_float(max(__float_as_uint(a) & 0x7F800000u, 0x3C800000u) + (20u << 23));
  return copysignf(__fsub_rn(__fadd_rn(a, K), K), x);
}
// pack two values already on the f16 grid (q8 outputs): the truncating conversion is exact
__device__ __forceinline__ uint32_t pkz(float a, float b) { return __builtin_bit_cast(uint32_t, __builtin_amdgcn_cvt_pkrtz(a, b)); }
// pk(q8(f16r(a)), q8(f16r(b))) from xb = pk(a, b), both halves at once on the f16 grid (q8h of swin_var / pre/post_native,
// proof q8h_cpu.py); finite inputs (an f16 overflow to inf still gives 448 like q8)
__device__ __forceinline__ uint32_t q8h(uint32_t xb) {
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
// to_e4m3 bytes (low byte of each half) of a q8h word: normal (|v| >= 2^-6) = ((bits - 0x2400) >> 7) + 8, which wraps
// to >= 448 below 2^-6; subnormal = |v| * 512 = bits(1024 + |v| * 512) - 0x6400, which is >= the normal code for
// normals; so the u16 min picks. Sign only for nonzero (to_e4m3(-0) = 0). Exhaustive check: perf_c32b src/q8chk.py.
__device__ __forceinline__ uint32_t e4h(uint32_t r) {
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
  const uint32_t nz = __builtin_bit_cast(uint32_t, ab + (u2){0x7FFF, 0x7FFF});   // bit 15 per half = nonzero
  return __builtin_bit_cast(uint32_t, __builtin_elementwise_min(nb, sb)) | ((r & nz & 0x80008000u) >> 8);
}
// 4 e4m3 bytes of memory channels m..m+3 from f32 values: reference channels pi32(m), +1, pi32(m+2), +1
__device__ __forceinline__ uint32_t e4x4(const float* o, int m) {
  const int c0 = pi32(m), c1 = pi32(m + 2);
  const uint32_t a = e4h(q8h(pk(o[c0], o[c0 + 1]))), b = e4h(q8h(pk(o[c1], o[c1 + 1])));
  return __builtin_amdgcn_perm(b, a, 0x06040200u);   // bytes a0, a2, b0, b2
}

// record offsets (C = 32)
struct Rec {
  int R_AFT, R_RS, R_QKV, R_BIAS, R_HS, R_OUT, R_ARS, R_DSRS;
  __device__ Rec(bool up) {
    constexpr int C = 32, R_CON = 4 * C * C;
    R_AFT = R_CON + 4 * C * C; R_RS = up ? R_AFT + 2 * C * C : R_AFT;
    R_QKV = R_RS + 2 * C + 32 + (up ? 2 * C : 0);
    R_BIAS = R_QKV + 3 * C * C; R_HS = R_BIAS + 8192; R_OUT = R_HS + 16;
    R_ARS = R_OUT + C * C; R_DSRS = R_ARS + 2 * C;
  }
};

extern "C" __global__ __launch_bounds__(64) __attribute__((amdgpu_waves_per_eu(WPE))) void k_swin_c32(VarParams P) {
  constexpr int C = 32;
  __shared__ uint32_t RA[2048];   // up: Pj f16 [16 px][32] | qn [16 i][64 tok] + kn [16 i][64 tok] | Pm / P [32 j][64 q] | ds: pooled f16 [16 px][32]
  __shared__ uint32_t RB[1056];   // Vt [32 key pair][33] half2 (keys 2j, 2j+1) per dim | ctx [16 kp][64 tok]

  const int t = threadIdx.x, lane = t & 31;
  const int wv = __builtin_amdgcn_readfirstlane(t >> 5);
  const cwp S = (cwp)P.scratch;
  const cfp Sf = (cfp)P.scratch;
  const int H = P.H, W = P.W, WT = W >> 2;
  const int r0 = P.Y0 + blockIdx.y * 8, c0 = P.X0 + blockIdx.x * 8;
  const bool up = P.flags & 8;
  // token t = (qy*2+qx)*16 + y*4 + x
  const int lr = ((t >> 5) & 1) * 4 + ((t >> 2) & 3), lc = ((t >> 4) & 1) * 4 + (t & 3);
  const int row = r0 + lr, col = c0 + lc, pix = (lr >> 1) * 4 + (lc >> 1);
  const bool in = row >= 0 && row < H && col >= 0 && col < W;

  // ---- the token's 32 input bytes (issued first: the upsample below overlaps their latency)
  uint32_t b[8] = {};
  if (in) {
    const int p = (row & 3) * 4 + (col & 3);
    if (P.flags & 1) {
      const uint8_t* src = P.in + (size_t)(row * W + col) * 16;
#pragma unroll
      for (int g = 0; g < 4; g++) { b[2 * g] = *(const uint32_t*)(src + 4 * g); b[2 * g + 1] = *(const uint32_t*)(src + (size_t)H * W * 16 + 4 * g); }
    } else {
      const uint8_t* src = P.in + (size_t)((row >> 2) * WT + (col >> 2)) * (C * 16) + (p & 7) * 64 + 4 * (p >> 3);
#pragma unroll
      for (int g = 0; g < 4; g++) { b[2 * g] = *(const uint32_t*)(src + 16 * g); b[2 * g + 1] = *(const uint32_t*)(src + 16 * g + 8); }
    }
  }

  // ---- upsample: Pj[16 pix][32] = f16(Wr . coarse), K = 64; lane px of wave wv makes outputs 16wv..+15
  if (up) {
    const int H2 = H >> 1, W2 = W >> 1, Hs = (H2 + 3) & ~3, Ws = (W2 + 3) & ~3;
    const int px = lane & 15, pr = (r0 >> 1) + (px >> 2), pc = (c0 >> 1) + (px & 3);
    uint32_t cb[16] = {};
    if (pr >= 0 && pr < H2 && pc >= 0 && pc < W2) {
#pragma unroll
      for (int ch = 0; ch < 2; ch++) {
        const uint8_t* src = P.p30 + (size_t)(ch * 2) * Hs * Ws * 16 + (size_t)(pr * Ws + pc) * 16;
#pragma unroll
        for (int g = 0; g < 4; g++) { cb[ch * 8 + 2 * g] = *(const uint32_t*)(src + 4 * g); cb[ch * 8 + 2 * g + 1] = *(const uint32_t*)(src + (size_t)Hs * Ws * 16 + 4 * g); }
      }
    }
    uint32_t cw[32];
#pragma unroll
    for (int ch = 0; ch < 2; ch++)
#pragma unroll
      for (int g = 0; g < 4; g++)
#pragma unroll
        for (int s2 = 0; s2 < 2; s2++)
#pragma unroll
          for (int hh = 0; hh < 2; hh++) {
            const int c = 16 * s2 + 4 * g + 2 * hh;
            cw[(ch * 32 + pi32(c)) >> 1] = x256(e4x2((cb[ch * 8 + 2 * g + s2] >> (16 * hh)) & 0xFFFFu));
          }
    float a[16] = {};
    const cwp wr = S + S_WR + wv * 512;
    gemv<32, 16>(a, cw, wr);
    _Float16* Pj = (_Float16*)RA;
    if (lane < 16)
#pragma unroll
      for (int n = 0; n < 16; n++) Pj[px * C + 16 * wv + n] = (_Float16)(a[n] * 256.f);
    __syncthreads();
  }

  // ---- load the token: X = matmul input (q8), R = MLP residual (x, or the f16 blend when up)
  uint32_t X[16], R[16];
  {
#pragma unroll
    for (int g = 0; g < 4; g++)
#pragma unroll
      for (int s = 0; s < 2; s++)
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          const int c = 16 * s + 4 * g + 2 * hh;
          X[pi32(c) >> 1] = x256(e4x2((b[2 * g + s] >> (16 * hh)) & 0xFFFFu));
        }
    if (up) {   // blend = f16(skip * gain + P(coarse)), zero outside the frame. skip * gain is exact in f32 and
                // skip * gain + P is exact in f64 (<= 49 significant bits), so one f32 rounding = the f64 reference.
#pragma unroll
      for (int k = 0; k < 16; k++) {
        const uint32_t v = X[k], pj = RA[pix * 16 + k];
        const float x0 = f16r(__fadd_rn(__fmul_rn(lo(v), Sf[S_UG + 2 * k]), lo(pj)));
        const float x1 = f16r(__fadd_rn(__fmul_rn(hi(v), Sf[S_UG + 2 * k + 1]), hi(pj)));
        R[k] = in ? pk(x0, x1) : 0u;
        X[k] = q8h(R[k]);
      }
    } else {
#pragma unroll
      for (int k = 0; k < 16; k++) R[k] = X[k];
    }
  }
  if (up) __syncthreads();   // Pj (RA) is read; qn/kn overwrite it below

  // ================= MLP in registers, rolled over hidden pairs (small code): per pair G, expand hidden 2G, 2G+1 over
  // K = 32, act + q8, then y += hd(G) . Wc[G] (the contract accumulates over G = 0..63 in order)
  float y[32] = {};
#pragma unroll 1
  for (int G = 0; G < 64; G += HPG) {
    const cwp we = S + S_WE + G * 32, wc = S + S_WC + G * 32;
    float acc[2 * HPG] = {};
#pragma unroll
    for (int kp = 0; kp < 16; kp++)
#pragma unroll
      for (int j = 0; j < 2 * HPG; j++) acc[j] = dot2(X[kp], we[j * 16 + kp], acc[j]);
    uint32_t hd[HPG];
#pragma unroll
    for (int g = 0; g < HPG; g++) hd[g] = pkz(q8f(act(acc[2 * g] * 256.f)), q8f(act(acc[2 * g + 1] * 256.f)));
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int g = 0; g < HPG; g++)
#pragma unroll
      for (int c = 0; c < 32; c++) y[c] = dot2(hd[g], wc[g * 32 + c], y[c]);
    __builtin_amdgcn_sched_barrier(0);
  }

  // ---- y = rs * x + mlp (f32, the attention skip); Yq = q8(f16(y))
  uint32_t Yq[16];
#pragma unroll
  for (int c = 0; c < 32; c++) y[c] = __fadd_rn(__fmul_rn((c & 1) ? hi(R[c >> 1]) : lo(R[c >> 1]), Sf[S_RSC + c]), y[c] * 256.f);
#pragma unroll
  for (int k = 0; k < 16; k++) Yq[k] = q8h(pk(y[2 * k], y[2 * k + 1]));

  // ================= attention (one head)
  uint32_t* qn = RA;          // [16 i][64 tok]
  uint32_t* kn = RA + 1024;   // [16 i][64 tok]
  uint32_t* Pm = RA;          // [32 key pair][64 q]
  uint32_t* Vt = RB;          // [32 key pair][33]
  uint32_t* ctx = RB;         // [16 kp][64 tok]
  {   // qkv in registers (K = 32), q and k L2-normalised in place (q with the head scale), v -> Vt
#pragma unroll 1
    for (int g = 0; g < 3; g++) {   // one gemv body for q, k, v (code size)
      float a[32] = {};
      gemv<16, 32>(a, Yq, S + S_WQ + g * 512);
#pragma unroll
      for (int n = 0; n < 32; n++) a[n] *= 256.f;
      if (g < 2) {
        float sc = norm_scale(a);
        if (g == 0) sc = __fmul_rn(sc, Sf[S_HSC]);
        uint32_t* dst = g ? kn : qn;
#pragma unroll
        for (int d = 0; d < 32; d += 2) dst[(d >> 1) * 64 + t] = pkz(q8f(__fmul_rn(a[d], sc)), q8f(__fmul_rn(a[d + 1], sc)));
      } else {
        uint32_t own[16];   // own[k] = v(dims 2k, 2k+1) as f16 pairs
#pragma unroll
        for (int k = 0; k < 16; k++) own[k] = q8h(pk(a[2 * k], a[2 * k + 1]));
        // the even token of a key pair writes dims 0..15, the odd one dims 16..31; each needs the partner's half
        const bool odd = t & 1;
        const int j = t >> 1;
#pragma unroll
        for (int k = 0; k < 8; k++) {
          const uint32_t send = odd ? own[k] : own[8 + k];
          const uint32_t recv = (uint32_t)__builtin_amdgcn_mov_dpp((int)send, 0xB1, 0xF, 0xF, false);   // quad_perm [1,0,3,2]: lane ^ 1
          const uint32_t mine = odd ? own[8 + k] : own[k];
          const uint32_t w0 = odd ? ((recv & 0xFFFFu) | (mine << 16)) : ((mine & 0xFFFFu) | (recv << 16));
          const uint32_t w1 = odd ? ((recv >> 16) | (mine & 0xFFFF0000u)) : ((mine >> 16) | (recv & 0xFFFF0000u));
          const int d = 2 * k + (odd ? 16 : 0);
          Vt[j * 33 + d] = w0; Vt[j * 33 + d + 1] = w1;
        }
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
    const float4* bp = (const float4*)(P.scratch + 4 * (S_BIAS + t * 64));
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
      h2 pr[4];
#pragma unroll
      for (int p = 0; p < 4; p++) { int b = 2 * p + m; pr[p] = ((pp[b] + pp[8 + b]) + pp[16 + b]) + pp[24 + b]; }
      hm[m] = ((pr[0] + pr[1]) + pr[2]) + pr[3];
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
  // ---- out_proj (lane = token), out = ars * y + Wo . ctx
  float o[32] = {};
  {
#pragma unroll 2
    for (int kp = 0; kp < 16; kp++) {   // rolled rows (code size), the ctx word straight from LDS
      const uint32_t cx = ctx[kp * 64 + t];
#pragma unroll
      for (int c = 0; c < 32; c++) o[c] = dot2(cx, S[S_WO + kp * 32 + c], o[c]);
      __builtin_amdgcn_sched_barrier(0);
    }
#pragma unroll
    for (int c = 0; c < 32; c++) o[c] = __fadd_rn(__fmul_rn(y[c], Sf[S_ARS + c]), o[c] * 256.f);
  }
  // ---- store q8(f16(out)) as e4m3: 32 bytes of the token
  if (in) {
    if (P.flags & 2) {
#pragma unroll
      for (int pl = 0; pl < 2; pl++) {
        uint32_t w[4];
#pragma unroll
        for (int k = 0; k < 4; k++) w[k] = e4x4(o, 16 * pl + 4 * k);
        *(uint4*)(P.out + (size_t)pl * H * W * 16 + (size_t)(row * W + col) * 16) = (uint4){w[0], w[1], w[2], w[3]};
      }
    } else {
      const int p = (row & 3) * 4 + (col & 3);
      uint8_t* dst = P.out + (size_t)((row >> 2) * WT + (col >> 2)) * (C * 16) + (p & 7) * 64 + 4 * (p >> 3);
#pragma unroll
      for (int a = 0; a < 4; a++)
#pragma unroll
        for (int b = 0; b < 2; b++) {
          *(uint32_t*)(dst + 16 * a + 8 * b) = e4x4(o, 4 * a + 16 * b);
        }
    }
  }
  if (!(P.flags & 4)) return;

  // ---- downsample: pooled = q8(pool2x2(out)) -> RA f16 [16 pix][32]; next = q8(f16(Wds . pooled)) -> +0x38 planes [4][H/2][W/2][16]
  {
    // RA (P) was last read before the ctx barrier
    const bool lead = !(t & 1) && !(t & 4);
    uint32_t* pw = RA;
#pragma unroll
    for (int c = 0; c < 32; c += 2) {   // x partner = lane ^ 1 (DPP quad_perm [1,0,3,2]), y partner = lane ^ 4 (DPP row_xmask:4)
      const float ox0 = __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(o[c]), 0xB1, 0xF, 0xF, false));
      const float ox1 = __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(o[c + 1]), 0xB1, 0xF, 0xF, false));
      const h2 h = (h2){(_Float16)(o[c] + ox0), (_Float16)(o[c + 1] + ox1)};
      const h2 hy = __builtin_bit_cast(h2, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, h), 0x164, 0xF, 0xF, false));
      const h2 tot = h + hy;
      if (lead) pw[pix * 16 + (c >> 1)] = pkz(q8f((float)tot.x * 0.25f), q8f((float)tot.y * 0.25f));
    }
  }
  __syncthreads();
  {
    const int px = lane & 15;
    float acc[32] = {};
    const cwp wd = S + S_WD + wv * 512;
#pragma unroll 2
    for (int kp = 0; kp < 16; kp++) {
      const uint32_t pw = RA[px * 16 + kp];
#pragma unroll
      for (int n = 0; n < 32; n++) acc[n] = dot2(pw, wd[kp * 32 + n], acc[n]);
      __builtin_amdgcn_sched_barrier(0);
    }
    const int H2 = H >> 1, W2 = W >> 1, Hs = (H2 + 3) & ~3, Ws = (W2 + 3) & ~3;
    const int pr = (r0 >> 1) + (px >> 2), pc = (c0 >> 1) + (px & 3);
    if (lane < 16 && pr >= 0 && pr < H2 && pc >= 0 && pc < W2) {
#pragma unroll
      for (int n = 0; n < 32; n++) acc[n] *= 256.f;
#pragma unroll
      for (int h = 0; h < 2; h++) {
        uint32_t w[4];
#pragma unroll
        for (int k = 0; k < 4; k++) w[k] = e4x4(acc, 16 * h + 4 * k);
        *(uint4*)(P.p38 + (size_t)(wv * 2 + h) * Hs * Ws * 16 + (size_t)(pr * Ws + pc) * 16) = (uint4){w[0], w[1], w[2], w[3]};
      }
    }
  }
}

// ---- prep (once per record, launched by the runtime at load with the step's kernarg, grid 1, 256 threads): stages the
// record's weights exactly as the 256-thread kernel did (put16 / stage) and stores them reordered into the module pool
// (layout above); leaves the shadow base in g_w16_last (0 = pool full), which the runtime writes to kernarg +0xa0.
constexpr size_t W16_POOL = 2u << 20;   // ponytail: 48.5 KB per record, 43 records; grow if a model has more c32 blocks
__device__ uint4 g_w16_pool[W16_POOL / 16];
__device__ unsigned long long g_w16_used, g_w16_last;
extern "C" __global__ __launch_bounds__(256) void k_swin_c32_prep(VarParams P) {
  constexpr int C = 32;
  __shared__ uint32_t L[2048];
  __shared__ unsigned long long base;
  const int tid = threadIdx.x;
  const uint8_t* rec = P.w;
  const bool up = P.flags & 8, ds = P.flags & 4;
  const Rec o(up);
  constexpr int R_CON = 4 * C * C;
  if (tid == 0) {
    const unsigned long long need = (4ull * S_END + 255) & ~255ull, b = atomicAdd(&g_w16_used, need);
    base = b + need <= W16_POOL ? b : ~0ull;
    g_w16_last = base == ~0ull ? 0 : (unsigned long long)((uint8_t*)g_w16_pool + base);
  }
  __syncthreads();
  if (base == ~0ull) return;
  uint32_t* S = (uint32_t*)((uint8_t*)g_w16_pool + base);
  float* Sf = (float*)S;
  for (int sc = 0; sc < 2; sc++) {   // expand chunk sc: L [16 kp][64 n]
    __syncthreads();
    if (tid < 128) put16(L, 64, *((const uint4*)(rec + sc * 4 * 512) + tid), tid & 31, (tid >> 5) * 16);
    __syncthreads();
    for (int i = tid; i < 1024; i += 256) { const int kp = i >> 6, nn = i & 63; S[S_WE + (64 * sc + nn) * 16 + kp] = L[i]; }
    __syncthreads();                 // contract chunk sc: L [32 kp][32 n]
    if (tid < 128) { const int i = tid >> 5; put16(L + (i >> 1) * 16 * 32, 32, *((const uint4*)(rec + R_CON + (sc * 4 + i) * 512) + (tid & 31)), tid & 31, (i & 1) * 16); }
    __syncthreads();
    for (int i = tid; i < 1024; i += 256) S[S_WC + sc * 1024 + i] = L[i];
  }
  __syncthreads();                   // qkv: L [16 kp][96 n]
  if (tid < 192) put16(L, 96, *((const uint4*)(rec + o.R_QKV) + tid), tid & 31, (tid >> 5) * 16);
  __syncthreads();
  for (int i = tid; i < 1536; i += 256) { const int kp = i / 96, n = i % 96; S[S_WQ + ((n >> 5) * 16 + kp) * 32 + (n & 31)] = L[i]; }
  __syncthreads();                   // out_proj: L [16 kp][32 n]
  stage(L, C, rec, 2, [&](int i, int& off, int& ncol, int& kpo) { off = o.R_OUT + i * 512; ncol = i * 16; kpo = 0; });
  __syncthreads();
  for (int i = tid; i < 512; i += 256) S[S_WO + i] = L[i];
  if (up) {                          // up resample Wr [32 n][64 k]: L [32 kp][32 n]
    __syncthreads();
    stage(L, 32, rec, 4, [&](int i, int& off, int& ncol, int& kpo) { off = o.R_AFT + i * 512; ncol = (i & 1) * 16; kpo = (i >> 1) * 16; });
    __syncthreads();
    for (int i = tid; i < 1024; i += 256) { const int kp = i >> 5, n = i & 31; S[S_WR + ((n >> 4) * 32 + kp) * 16 + (n & 15)] = L[i]; }
  }
  if (ds) {                          // ds resample: L [16 kp][64 n]
    __syncthreads();
    stage(L, 2 * C, rec, 4, [&](int i, int& off, int& ncol, int& kpo) { off = o.R_DSRS + i * 512; ncol = i * 16; kpo = 0; });
    __syncthreads();
    for (int i = tid; i < 1024; i += 256) { const int kp = i >> 6, n = i & 63; S[S_WD + ((n >> 5) * 16 + kp) * 32 + (n & 31)] = L[i]; }
  }
  if (tid < 32) {
    const _Float16* rs = (const _Float16*)(rec + o.R_RS);
    const uint4 rs8 = *(const uint4*)rs;
    const int rso = (rs8.x | rs8.y | rs8.z | rs8.w) == 0 ? 8 : 0;
    Sf[S_RSC + tid] = (float)rs[rso + tid];
    Sf[S_ARS + tid] = (float)((const _Float16*)(rec + o.R_ARS))[tid];
    Sf[S_UG + tid] = up ? (float)((const _Float16*)(rec + o.R_RS + 2 * C + 32))[tid] : 0.f;
    if (tid == 0) Sf[S_HSC] = ((const float*)(rec + o.R_HS))[0];
  }
  const _Float16* braw = (const _Float16*)(rec + o.R_BIAS);
  for (int e = tid; e < 4096; e += 256) {   // logits thread th = qb + 8kb holds queries 8qb + a, keys 8kb + b
    const int th = e >> 6, a = (e >> 3) & 7, b = e & 7, q = 8 * (th & 7) + a, s = 8 * (th >> 3) + b;
    const int ii = q & 15, j = s & 15;
    const int lane = 4 * (ii & 7) + ((j & 7) >> 1);
    const int slot = (j & 1) | ((ii >> 3) << 1) | ((j >> 3) << 2);
    Sf[S_BIAS + e] = fmaf((float)braw[(4 * (q >> 4) + (s >> 4)) * 256 + lane * 8 + slot], 0.044921875f, 1.30078125f);
  }
}

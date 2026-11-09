// Native replacements for the last six manifest kernels that ran from translated code (native_io.co):
//   k_import_n           k_import       game colour RGBA16F -> padded RGB32F (mode 0: raw, mirrored padding)
//   k_export_n           k_export       network RGBA32F -> RGBA16F frame (mode 0)
//   k_repack_n           k_repack       4x4-tiled map <-> 16-token tiles (ViT entry / exit)
//   k_attention_n        k_attention2   ViT global attention (dlssnr-pytorch vit.VitAttention / attention._vit_kernel)
//   k_final_head_n       k_final_head   split_swin.SplitFinalHead: q8(f16(W x)), 512 -> 1024
//   k_conv_res_views_n   k_conv_res_views  split_swin.SplitProjPool: projection + residual, plus the 2x2-mean copy
// Math from the MIT reference (E:\dlss 5\dlssnr-pytorch) and measured input -> output behaviour (kcmp.cpp); kernarg
// layouts are the manifest's, so the overrides copy the original kernarg unchanged. Layout helpers: nk_common.h.
#include "nk_common_io.h"   // io_head keeps its own copy (no nr_params.h; it defines its own kernarg structs)

// ---------------------------------------------------------------------------------------------------------------
// k_import (mode 0, format 0 = RGBA16F): out[y][x] = rgb(src[m(y)][m(x)]) as f32 over the padded extent; the padding
// mirrors the frame about its last row / column (io.frame_to_tensor). grid (ceil(PW/256), PH).
struct ImportParams { const uint8_t* src; int pitch, fmt, H, W, PH, PW; float* out; int mode; float scale; };
__device__ __forceinline__ int mirror(int i, int n) { const int r = i < n ? i : 2 * n - i - 2; return r < 0 ? 0 : r > n - 1 ? n - 1 : r; }

extern "C" __global__ void __launch_bounds__(256) k_import_n(ImportParams P) {
  const int x = blockIdx.x * 256 + threadIdx.x, y = blockIdx.y;
  if (x >= P.PW || y >= P.PH) return;
  const f16* s = (const f16*)(P.src + (size_t)mirror(y, P.H) * P.pitch) + 4 * mirror(x, P.W);
  float* o = P.out + ((size_t)y * P.PW + x) * 3;
  o[0] = (float)s[0]; o[1] = (float)s[1]; o[2] = (float)s[2];
}

// ---------------------------------------------------------------------------------------------------------------
// k_export (mode 0): dst RGBA16F = f16(net RGBA32F), net row stride in pixels. grid (ceil(W/256), H).
struct ExportParams { const float* net; int stride, H, W, pitch, fmt; uint8_t* dst; };
extern "C" __global__ void __launch_bounds__(256) k_export_n(ExportParams P) {
  const int x = blockIdx.x * 256 + threadIdx.x, y = blockIdx.y;
  if (x >= P.W || y >= P.H) return;
  const float4 v = *(const float4*)(P.net + ((size_t)y * P.stride + x) * 4);
  f16* d = (f16*)(P.dst + (size_t)y * P.pitch) + 4 * x;
  d[0] = (f16)v.x; d[1] = (f16)v.y; d[2] = (f16)v.z; d[3] = (f16)v.w;
}

// ---------------------------------------------------------------------------------------------------------------
// k_repack, C = 1024: pixel (y, x) of an H x W map in 4x4 tiles (ceil(W/4) tile columns) <-> token y*W + x in
// 16-token tiles. Both tiles keep the same in-tile byte layout, so a token moves as 64 16-byte pieces
// (piece = 4 channel bytes x 4 positions; position p sits at tok_off(p)). fwd = 1: map -> tokens, 0: back.
// Padding (map rows/cols past H/W, tokens past H*W) is left untouched. grid (256): grid-stride over all pieces.
struct RepackParams { const uint8_t* in; uint8_t* out; int H, W, ntok, fwd; };
extern "C" __global__ void __launch_bounds__(256) k_repack_n(RepackParams P) {
  const int n = P.H * P.W, TW = (P.W + 3) >> 2;
  for (int i = blockIdx.x * 256 + threadIdx.x; i < n * 256; i += gridDim.x * 256) {   // i = token * 256 + dword
    const int t = i >> 8, d = i & 255, y = t / P.W, x = t % P.W;
    const int mt = (y >> 2) * TW + (x >> 2), mp = (y & 3) * 4 + (x & 3);
    const int r = (d >> 3) * 32 + ((d & 3) << 2) + ((d >> 2) & 1) * 16;   // storage channel of this dword (r % 4 == 0)
    const size_t mo = (size_t)mt * 16384 + ch_off(r) + tok_off(mp), to = (size_t)(t >> 4) * 16384 + ch_off(r) + tok_off(t & 15);
    if (P.fwd) *(uint32_t*)(P.out + to) = *(const uint32_t*)(P.in + mo);
    else *(uint32_t*)(P.out + mo) = *(const uint32_t*)(P.in + to);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// GEMM core shared with native_c512 / native_vit (same staging, same dot2 order): 64 tokens x 64 storage rows per
// workgroup, thread (ty = tid>>4, tx = tid&15) owns tokens 4ty..+3 and rows n0+4tx..+3.
struct WPiece { const uint8_t* src; size_t step; int wn[4], wk[4]; };
template <int K, int INNER>
__device__ __forceinline__ WPiece wpiece_std(const uint8_t* Wr, int n0) {
  const int tid = threadIdx.x, q = tid & 31, ntl = tid >> 5 & 3, kgl = tid >> 7, nt = (n0 >> 4) + ntl;
  WPiece w;
  w.src = Wr + (size_t)(((nt / INNER) * (K / 32) + kgl) * INNER + nt % INNER) * 512 + q * 16;
  w.step = (size_t)2 * INNER * 512;
#pragma unroll
  for (int e = 0; e < 4; e++) {
    const int n = nt * 16 + (e >> 1) * 8 + (q >> 2);
    w.wn[e] = ((n & ~31) | ipi32(n & 31)) - n0;
    w.wk[e] = kgl * 8 + (e & 1) + 2 * (q & 3);
  }
  return w;
}
// The split-Swin projections of these two kernels keep their accumulator in f16: each 32-channel slice of K is summed
// in f32 (exact for e4m3 products) and then added to the f16 accumulator (measured: 0 of 163840 values differ).
__device__ __forceinline__ void mma16(const uint2* As, const uint2* Ws, float acc[4][4]) {
  const int ty = threadIdx.x >> 4, tx = threadIdx.x & 15;
#pragma unroll
  for (int half = 0; half < 2; half++) {
    float s[4][4] = {};
#pragma unroll 4
    for (int kd = half * 8; kd < half * 8 + 8; kd++) {
      const uint4 a01 = *(const uint4*)(As + kd * 64 + ty * 4), a23 = *(const uint4*)(As + kd * 64 + ty * 4 + 2);
      const uint4 w01 = *(const uint4*)(Ws + kd * 64 + tx * 4), w23 = *(const uint4*)(Ws + kd * 64 + tx * 4 + 2);
      const uint32_t A[4][2] = {{a01.x, a01.y}, {a01.z, a01.w}, {a23.x, a23.y}, {a23.z, a23.w}};
      const uint32_t B[4][2] = {{w01.x, w01.y}, {w01.z, w01.w}, {w23.x, w23.y}, {w23.z, w23.w}};
#pragma unroll
      for (int i = 0; i < 4; i++)
#pragma unroll
        for (int e = 0; e < 4; e++) s[i][e] = dot2(A[i][1], B[e][1], dot2(A[i][0], B[e][0], s[i][e]));
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
      for (int e = 0; e < 4; e++) acc[i][e] = (float)(f16)__fadd_rn(acc[i][e], s[i][e] * 65536.f);   // acc in real units
  }
}
template <int K>
__device__ __forceinline__ void gemm64(const uint8_t* __restrict__ X, int tile0, int T, const WPiece& wp, float acc[4][4], uint2* lds) {
  constexpr int NC = K / 64;
  const int tid = threadIdx.x, tt = tid >> 6, sa = (tid >> 5) & 1, q = tid & 31, atile = tile0 + tt < T ? tile0 + tt : -1;
  auto loadA = [&](int c) -> uint4 {
    if (atile < 0) return make_uint4(0, 0, 0, 0);
    return *(const uint4*)(X + (size_t)atile * (K * 16) + sa * 512 + q * 16 + (size_t)c * 1024);
  };
  int am[4], ak[4];
#pragma unroll
  for (int e = 0; e < 4; e++) { am[e] = tt * 16 + (q >> 2) + 8 * (e & 1); ak[e] = sa * 8 + 2 * (q & 3) + (e >> 1); }
  uint4 ra = loadA(0), rw = *(const uint4*)wp.src;
  for (int c = 0; c < NC; c++) {
    uint2* As = lds + (c & 1) * 2048;
    uint2* Ws = As + 1024;
    const uint32_t av[4] = {ra.x, ra.y, ra.z, ra.w}, wv[4] = {rw.x, rw.y, rw.z, rw.w};
#pragma unroll
    for (int e = 0; e < 4; e++) { As[ak[e] * 64 + am[e]] = cvt4(av[e]); Ws[wp.wk[e] * 64 + wp.wn[e]] = cvt4(wv[e]); }
    __syncthreads();
    if (c + 1 < NC) { ra = loadA(c + 1); rw = *(const uint4*)(wp.src + (size_t)(c + 1) * wp.step); }
    mma16(As, Ws, acc);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// k_final_head: pooled 512-channel map (T tiles) -> 1024 channels, q8(f16(W x)), no residual.
// Record: weight [1024][512], 16x32 slices with INNER 64. grid (T, 16): x-blocks past ceil(T/4) exit.
struct HeadParams { const uint8_t* x; uint8_t* out; const uint8_t* w; int T; };
extern "C" __global__ void __launch_bounds__(256) k_final_head_n(HeadParams P) {
  __shared__ uint2 lds[4096];
  const int tile0 = blockIdx.x * 4, n0 = blockIdx.y * 64;
  if (tile0 >= P.T || n0 >= 1024) return;
  float acc[4][4] = {};
  gemm64<512>(P.x, tile0, P.T, wpiece_std<512, 64>(P.w, n0), acc, lds);
  const int ty = threadIdx.x >> 4, r = n0 + (threadIdx.x & 15) * 4;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int m = ty * 4 + i, tile = tile0 + (m >> 4);
    if (tile >= P.T) continue;
    uint32_t o = 0;
#pragma unroll
    for (int e = 0; e < 4; e++) o |= e4m3_byte(acc[i][e]) << (8 * e);
    *(uint32_t*)(P.out + (size_t)tile * 16384 + ch_off(r) + tok_off(m & 15)) = o;
  }
}

// ---------------------------------------------------------------------------------------------------------------
// k_conv_res_views (block 30 SplitProjPool): v = f16 accumulation of W x onto f16(gain * res) per pixel, out = e4m3(v) and
// pooled = e4m3(f16(f16(f16(v00 + v01) + f16(v10 + v11)) * 0.25)) on the half-resolution map (H2 x W2, 4x4 tiles).
// Same record as k_conv_res2 (weight [512][512] INNER 32, skip_weight f16[512]). grid (T, 8): x-blocks past ceil(T/4) exit.
struct ConvPlParams { const uint8_t* x; const uint8_t* unused0; const uint8_t* res; uint8_t* out; const uint8_t* unused1;
                      const uint8_t* w; int H, W; uint8_t* pooled; int H2, W2, T; };
extern "C" __global__ void __launch_bounds__(256) k_conv_res_views_n(ConvPlParams P) {
  __shared__ uint2 lds[4096];
  const int tile0 = blockIdx.x * 4, n0 = blockIdx.y * 64;
  if (tile0 >= P.T || n0 >= 512) return;
  const int ty = threadIdx.x >> 4, r = n0 + (threadIdx.x & 15) * 4, WT = P.W >> 2, PWT = (P.W2 + 3) >> 2;
  const int tile = tile0 + (ty >> 2), row = ty & 3;   // this thread: one row of 4 pixels of one tile
  const bool live = tile < P.T;
  const f16* gain = (const f16*)(P.w + 512 * 512);
  float acc[4][4] = {};   // the f16 accumulator starts at f16(gain * res) (measured: 0 of 327680 values differ)
  if (live) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const uint32_t rd = *(const uint32_t*)(P.res + (size_t)tile * 8192 + ch_off(r) + tok_off(row * 4 + i));
#pragma unroll
      for (int e = 0; e < 4; e++) acc[i][e] = (float)(f16)__fmul_rn(e4f(rd >> (8 * e)), (float)gain[(r & ~31) | pi32((r + e) & 31)]);
    }
  }
  gemm64<512>(P.x, tile0, P.T, wpiece_std<512, 32>(P.w, n0), acc, lds);
  f16 v[4][4];
  if (live) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      uint32_t ob = 0;
#pragma unroll
      for (int e = 0; e < 4; e++) { v[i][e] = (f16)acc[i][e]; ob |= e4m3_byte(acc[i][e]) << (8 * e); }
      *(uint32_t*)(P.out + (size_t)tile * 8192 + ch_off(r) + tok_off(row * 4 + i)) = ob;
    }
  }
  // 2x2 mean: columns pair inside the thread, rows pair with the thread of the neighbouring row (lane ^ 16)
  f16 h[2][4];
#pragma unroll
  for (int j = 0; j < 2; j++)
#pragma unroll
    for (int e = 0; e < 4; e++) h[j][e] = live ? (f16)(v[2 * j][e] + v[2 * j + 1][e]) : (f16)0.f;
  f16 pv[2][4];
#pragma unroll
  for (int j = 0; j < 2; j++)
#pragma unroll
    for (int e = 0; e < 4; e++) pv[j][e] = (f16)__shfl_xor((float)h[j][e], 16, 32);
  if (!live || (row & 1)) return;
  const int Y = (tile / WT) * 2 + (row >> 1);
#pragma unroll
  for (int j = 0; j < 2; j++) {
    const int X = (tile % WT) * 2 + j;
    if (Y >= P.H2 || X >= P.W2) continue;
    uint32_t ob = 0;
#pragma unroll
    for (int e = 0; e < 4; e++) ob |= e4m3_byte((float)((f16)(h[j][e] + pv[j][e]) * (f16)0.25f)) << (8 * e);
    *(uint32_t*)(P.pooled + (size_t)((Y >> 2) * PWT + (X >> 2)) * 8192 + ch_off(r) + tok_off((Y & 3) * 4 + (X & 3))) = ob;
  }
}


// ---------------------------------------------------------------------------------------------------------------
// k_attention2: global attention over ntok tokens (n valid, the rest zero padding), 32 heads of 32 channels.
// p = approx_exp(f16(q . k)): the f16 ramp fma(x, 0.0895, 1.709) clamped to [1.4395, 1.9775], its 10 fraction bits
// shifted up 4 read as f16. Denominator in f16 in the network's order: per 16-key block p[c] + p[c+8], the 4 blocks
// of a 64-key group accumulated, the 8 lanes as ((a0+a2)+a4)+a6 + ((a1+a3)+a5)+a7, groups added in turn, then the
// padding keys' exp(0) sum subtracted. Numerator: f16 accumulator, each 32 keys' e4m3(p) * v summed in f32 and added
// (as the split-Swin projections). out = e4m3(f16(num * f16(1/den))).
// Everything runs in storage channel order (a head is one 32-channel storage group; the permutation inside it
// does not change any result). grid (ntok/64, 32): workgroup = 64 queries x 1 head. Per 64-key group, thread
// (query ql, quarter dg) forms the logits of keys 16dg..16dg+15, then the numerator of channels 8dg..8dg+7 over all 64.
struct AttnParams1d { const uint8_t* q; const uint8_t* k; const uint8_t* v; uint8_t* out; int ntok, n; };

__device__ __forceinline__ f16 vit_exp(float logit) {
  const f16 y = __builtin_fmaf16((f16)logit, (f16)0.08953857421875f, (f16)1.708984375f);
  const f16 c = y < (f16)1.439453125f ? (f16)1.439453125f : y > (f16)1.9775390625f ? (f16)1.9775390625f : y;
  return __builtin_bit_cast(f16, (uint16_t)((__builtin_bit_cast(uint16_t, c) & 0x3FF) << 4));
}
__device__ __forceinline__ size_t tok_byte(int t, int r) { return (size_t)(t >> 4) * 16384 + ch_off(r & ~3) + tok_off(t & 15) + (r & 3); }

#ifdef FAST_ATTN
// v_dot2 version (not bit-exact: pairs of products per add). Q and K as half2 of e4m3 * 2^-8 (exact), logits = dot2 sum * 2^16;
// V transposed per channel over key pairs, P = q8(p) as f16, numerator = dot2 sum * 2^8.
extern "C" __global__ void __launch_bounds__(256) k_attention_n(AttnParams1d P) {
  __shared__ uint32_t ks[64][17];    // key x 16 channel pairs (+1 pad)
  __shared__ uint16_t vt[32][66];    // channel x key (f16 bits, e4m3 * 2^-8)
  __shared__ f16 ps[64][66];
  const int tid = threadIdx.x, ql = tid & 63, dg = tid >> 6, h = blockIdx.y, qi = blockIdx.x * 64 + ql;
  uint32_t q[16];
#pragma unroll
  for (int i = 0; i < 16; i++) q[i] = qi < P.n ? e4x2((uint32_t)P.q[tok_byte(qi, h * 32 + 2 * i)] | ((uint32_t)P.q[tok_byte(qi, h * 32 + 2 * i + 1)] << 8)) : 0u;
  float num[8] = {};
  f16 den = 0;
  uint32_t kw[2], vw[2];
  const auto fetch = [&](int g) {
#pragma unroll
    for (int u = 0; u < 2; u++) {
      const int i = tid + 256 * u, key = g * 64 + (i >> 3);
      const size_t o = tok_byte(key, h * 32 + (i & 7) * 4);
      kw[u] = key < P.ntok ? *(const uint32_t*)(P.k + o) : 0u;
      vw[u] = key < P.ntok ? *(const uint32_t*)(P.v + o) : 0u;
    }
  };
  fetch(0);
  for (int g = 0; g * 64 < P.ntok; g++) {
    __syncthreads();
#pragma unroll
    for (int u = 0; u < 2; u++) {
      const int i = tid + 256 * u, key = i >> 3, c0 = (i & 7) * 4;
      const uint2 kk = cvt4(kw[u]), vv = cvt4(vw[u]);
      ks[key][c0 / 2] = kk.x; ks[key][c0 / 2 + 1] = kk.y;
      vt[c0][key] = (uint16_t)vv.x; vt[c0 + 1][key] = (uint16_t)(vv.x >> 16); vt[c0 + 2][key] = (uint16_t)vv.y; vt[c0 + 3][key] = (uint16_t)(vv.y >> 16);
    }
    __syncthreads();
    if ((g + 1) * 64 < P.ntok) fetch(g + 1);
#pragma unroll 4
    for (int c = 0; c < 16; c++) {
      const uint32_t* kr = ks[dg * 16 + c];
      float s = 0.f;
#pragma unroll
      for (int i = 0; i < 16; i++) s = dot2(q[i], kr[i], s);
      ps[ql][dg * 16 + c] = vit_exp(s * 65536.f);
    }
    __syncthreads();
#pragma unroll
    for (int half = 0; half < 2; half++) {
      float s[8] = {};
      for (int cp = half * 16; cp < half * 16 + 16; cp++) {
        h2 pq; pq.x = (f16)q8((float)ps[ql][2 * cp]); pq.y = (f16)q8((float)ps[ql][2 * cp + 1]);
        const uint32_t pb = __builtin_bit_cast(uint32_t, pq);
#pragma unroll
        for (int j = 0; j < 8; j++) s[j] = dot2(pb, *(const uint32_t*)&vt[dg * 8 + j][2 * cp], s[j]);
      }
#pragma unroll
      for (int j = 0; j < 8; j++) num[j] = (float)(f16)__fadd_rn(num[j], s[j] * 256.f);
    }
    f16 acc[8];
#pragma unroll
    for (int b = 0; b < 4; b++)
#pragma unroll
      for (int c = 0; c < 8; c++) { const f16 pr = ps[ql][b * 16 + c] + ps[ql][b * 16 + c + 8]; acc[c] = b ? (f16)(acc[c] + pr) : pr; }
    const f16 ev = (f16)((f16)((f16)(acc[0] + acc[2]) + acc[4]) + acc[6]), od = (f16)((f16)((f16)(acc[1] + acc[3]) + acc[5]) + acc[7]);
    const f16 gs = ev + od;
    den = g ? (f16)(den + gs) : gs;
  }
  if (qi >= P.n) return;
  const f16 pad = (f16)((float)(P.ntok - P.n) * (float)vit_exp(0.f));
  const f16 inv = (f16)__fdiv_rn(1.f, fmaxf((float)(f16)(den - pad), 6.198883056640625e-05f));
#pragma unroll
  for (int j = 0; j < 8; j++) P.out[tok_byte(qi, h * 32 + dg * 8 + j)] = (uint8_t)e4m3_byte((float)((f16)num[j] * inv));
}
#else
extern "C" __global__ void __launch_bounds__(256) k_attention_n(AttnParams1d P) {
  __shared__ float ks[64][32], vs[64][32];
  __shared__ f16 ps[64][66];   // p[query][key] of the current group (row padded against bank conflicts)
  const int tid = threadIdx.x, ql = tid & 63, dg = tid >> 6, h = blockIdx.y, qi = blockIdx.x * 64 + ql;
  float q[32];
#pragma unroll
  for (int d = 0; d < 32; d++) q[d] = qi < P.n ? e4f(P.q[tok_byte(qi, h * 32 + d)]) : 0.f;
  float num[8] = {};
  f16 den = 0;
  // K/V staging: 4 channel bytes per load (channels r..r+3 are contiguous), and the next group's words are fetched into
  // registers before this group's math, so the global latency hides behind it (same values as byte loads: bit-identical)
  uint32_t kw[2], vw[2];
  const auto fetch = [&](int g) {
#pragma unroll
    for (int u = 0; u < 2; u++) {   // i = tid + 256 u = key * 8 + 4-channel chunk
      const int i = tid + 256 * u, key = g * 64 + (i >> 3);
      const size_t o = tok_byte(key, h * 32 + (i & 7) * 4);
      kw[u] = key < P.ntok ? *(const uint32_t*)(P.k + o) : 0u;
      vw[u] = key < P.ntok ? *(const uint32_t*)(P.v + o) : 0u;
    }
  };
  fetch(0);
  for (int g = 0; g * 64 < P.ntok; g++) {
    __syncthreads();
#pragma unroll
    for (int u = 0; u < 2; u++) {
      const int i = tid + 256 * u, c0 = (i & 7) * 4;
#pragma unroll
      for (int b = 0; b < 4; b++) { ks[i >> 3][c0 + b] = e4f(kw[u] >> 8 * b); vs[i >> 3][c0 + b] = e4f(vw[u] >> 8 * b); }
    }
    __syncthreads();
    if ((g + 1) * 64 < P.ntok) fetch(g + 1);
#pragma unroll 4
    for (int c = 0; c < 16; c++) {
      const float* kr = ks[dg * 16 + c];
      float s = 0.f;
#pragma unroll
      for (int d = 0; d < 32; d++) s = __fadd_rn(s, __fmul_rn(q[d], kr[d]));
      ps[ql][dg * 16 + c] = vit_exp(s);
    }
    __syncthreads();
#pragma unroll
    for (int half = 0; half < 2; half++) {
      float s[8] = {};
      for (int c = half * 32; c < half * 32 + 32; c++) {
        const float pq = q8((float)ps[ql][c]);
        const float* vr = vs[c] + dg * 8;
#pragma unroll
        for (int j = 0; j < 8; j++) s[j] = __fadd_rn(s[j], __fmul_rn(pq, vr[j]));
      }
#pragma unroll
      for (int j = 0; j < 8; j++) num[j] = (float)(f16)__fadd_rn(num[j], s[j]);
    }
    f16 acc[8];
#pragma unroll
    for (int b = 0; b < 4; b++)
#pragma unroll
      for (int c = 0; c < 8; c++) { const f16 pr = ps[ql][b * 16 + c] + ps[ql][b * 16 + c + 8]; acc[c] = b ? (f16)(acc[c] + pr) : pr; }
    const f16 ev = (f16)((f16)((f16)(acc[0] + acc[2]) + acc[4]) + acc[6]), od = (f16)((f16)((f16)(acc[1] + acc[3]) + acc[5]) + acc[7]);
    const f16 gs = ev + od;
    den = g ? (f16)(den + gs) : gs;
  }
  if (qi >= P.n) return;
  const f16 pad = (f16)((float)(P.ntok - P.n) * (float)vit_exp(0.f));
  const f16 inv = (f16)__fdiv_rn(1.f, fmaxf((float)(f16)(den - pad), 6.198883056640625e-05f));
#pragma unroll
  for (int j = 0; j < 8; j++) P.out[tok_byte(qi, h * 32 + dg * 8 + j)] = (uint8_t)e4m3_byte((float)((f16)num[j] * inv));
}
#endif

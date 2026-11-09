// MLP weight staging shared by k_pre_native / k_post_native (needs W_EXP, W_CON, h2).
// One hidden chunk hc (8 of 128) uses 1 KB of weights: expand rows hc..hc+7 (contiguous 512 B) and contract
// columns hc..hc+7 of the 32 rows (16 B each). Lane t (0..63) fetches one 16 B piece with one dwordx4 load,
// the workgroup writes it to LDS once, and both waves read it back as broadcasts. Values and the order they
// enter each dot2 are unchanged; only the path into the registers differs (proof: prew_check.py).
// LDS chunk layout (h2 units): expand (j, k) at j*16 + k, contract (c, j) at 128 + c*4 + j.
#ifndef PREW
#define PREW 1
#endif
template <class WP>
__device__ __forceinline__ uint4 wfetch(WP Wt, int hc, int t) {
  const int i = t < 32 ? (W_EXP + hc * 32) / 2 + 4 * t : (W_CON + (t - 32) * 128 + hc) / 2;
  return *(const uint4*)(Wt + i);
}
#define WEXP_L(L, j, k) (L)[(j) * 16 + (k)]
#define WCON_L(L, c, j) (L)[128 + (c) * 4 + (j)]
typedef __attribute__((address_space(3))) const h2* LP;
// Ties a row's LDS reads to the end of the previous row (volatile asms keep their order): without it the compiler
// hoists a whole chunk's 64 reads (256 VGPRs) to the top of the loop body and spills.
__device__ __forceinline__ LP prew_pin(LP p, float done) { asm volatile("" : "+v"(p) : "v"(done)); return (LP)__builtin_assume_aligned(p, 16); }
#if PREW
#define PREW_ROW() Lr = prew_pin(Lc, pin)
#define PREW_DONE(v) pin = (v)
#else
#define PREW_ROW() ((void)0)
#define PREW_DONE(v) ((void)0)
#endif
// LDS-only workgroup barrier: __syncthreads() also emits buffer_gl0_inv (WGP mode), dropping from L0 the weight
// lines the other waves' loads of the same chunk want
#define PREW_SYNC() asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory")

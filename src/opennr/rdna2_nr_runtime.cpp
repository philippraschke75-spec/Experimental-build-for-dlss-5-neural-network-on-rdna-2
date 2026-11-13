// Rdna2NrRuntime.dll - the RDNA2 DLSS-NR port (nr_runtime.dll, HIP) behind the lmxxf runtime C ABI (LmxxfNrApi.h, ABI v1),
// so AMDNR (3zwr1's OptiScaler fork) can drive it on RX 6000 exactly like LmxxfNrRuntime.dll.
// Copyright (c) 2026 the gfx1030-dlss-nr-translation-lab authors. GPL-3.0-or-later. LmxxfNrApi.h: MIT, lmxxf / Kien (see NOTICE).
//
// Frame flow (host = AMDNR, see its LmxxfBackend.h):
//   PrepareFrame  : picks/builds the network for the colour size (nr_data\<model id>\<W>x<H>, loaded on a thread; until it is
//                   ready every job is a passthrough job), hands out private_output (RGBA16F, colour size, NON_PIXEL_SHADER_RESOURCE).
//   RecordInputs  : into the host's list: colour -> private copy (base), colour -> network input in the shared io buffer (+ the
//                   warped history), WriteBufferImmediate(flag_in = seq). Same pieces as dll/dlssnr_nr.cpp's sync path.
//   EnqueueHipAsync: nr_ext_enqueue2 - the runtime's own thread waits (bounded) for flag_in, runs the network (and the chained
//                   extra passes), writes the output into io and flag_out = seq. No D3D12 queue wait anywhere.
//   OutputReady   : nr_ext_status, never waits.
//   RecordOutputs : if the CPU already saw the job finish: io -> private_output (base + edit); else a bounded one-thread GPU spin on
//                   flag_out first (TDR-safe, like dlssnr_nr.cpp), passthrough on timeout.
// No frame skipping here: the host interleaves. nr_split (ini, default 0) is the only exception, see Rdna2NrRuntime.ini.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#define LMXXF_NR_RUNTIME_EXPORTS
#include <cmath>
#include "LmxxfNrApi.h"
#include <limits>
#undef NAN   // older MSVC + newer UCRT headers: NAN expands to __builtin_nanf, which cl 19.16 lacks
#define NAN (std::numeric_limits<float>::quiet_NaN())
#include "../runtime/weights_ht.h"
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

// ---------------------------------------------------------------- log, error, settings
static std::mutex g_logMu, g_errMu; static FILE* g_log; static std::string g_err, g_dir; static std::atomic<uint64_t> g_imports{0};
static void logf(const char* fmt, ...) {
    std::lock_guard<std::mutex> g(g_logMu); if (!g_log) return;
    va_list a; va_start(a, fmt); vfprintf(g_log, fmt, a); va_end(a); fputc('\n', g_log); fflush(g_log);
}
static int fail(int code, const char* fmt, ...) {   // sets GetLastError's text, logs it, returns code
    char b[512]; va_list a; va_start(a, fmt); vsnprintf(b, sizeof b, fmt, a); va_end(a);
    { std::lock_guard<std::mutex> g(g_errMu); g_err = b; } logf("error: %s", b); return code;
}
static void clearErr() { std::lock_guard<std::mutex> g(g_errMu); g_err.clear(); }

struct Settings {
    std::string dataDir, model = "auto", modelRes = "auto", weightsDll; int vitEvery = 0, reuseBn = 1, hist = 1, split = 0, enc = 2, spinMs = 250, allowAny = 0;
    float hLo = 0.08f, hHi = 0.25f, hdrFrac = 0.125f;
};
static Settings g_set; static std::once_flag g_setOnce;
static std::string ini(const char* k, const char* def) { char b[1024]; GetPrivateProfileStringA("nr", k, def, b, sizeof b, (g_dir + "Rdna2NrRuntime.ini").c_str()); return b; }
static void loadSettings() {
    Settings& s = g_set;
    s.dataDir = ini("data_dir", "nr_data"); if (s.dataDir.size() < 2 || (s.dataDir[1] != ':' && s.dataDir[0] != '\\' && s.dataDir[0] != '/')) s.dataDir = g_dir + s.dataDir;
    s.model = ini("model", "auto"); s.modelRes = ini("model_res", "auto"); s.weightsDll = ini("weights_dll", "");
    s.vitEvery = std::max(0, atoi(ini("vit_every", "0").c_str())); s.reuseBn = _stricmp(ini("reuse", "bottleneck").c_str(), "vit") != 0;
    s.hist = std::clamp(atoi(ini("history", "1").c_str()), 0, 2);
    s.split = atoi(ini("nr_split", "0").c_str()); s.split = s.split <= 1 ? 0 : std::min(s.split, 8);
    const std::string e = ini("encode", "auto"); s.enc = !_stricmp(e.c_str(), "ldr") ? 0 : !_stricmp(e.c_str(), "hdr") ? 1 : 2;
    s.spinMs = std::clamp(atoi(ini("sync_timeout_ms", "250").c_str()), 10, 1000); s.allowAny = atoi(ini("allow_any_gpu", "0").c_str()) != 0;
    auto f = [](const char* k, const char* d, float lo, float hi) { float v = (float)atof(ini(k, d).c_str()); return v == v ? std::clamp(v, lo, hi) : (float)atof(d); };
    s.hLo = f("history_gate_lo", "0.08", 0.f, 1.f); s.hHi = std::max(f("history_gate_hi", "0.25", 0.f, 2.f), s.hLo + 1e-4f); s.hdrFrac = f("hdr_auto_frac", "0.125", 0.001f, 1.f);
    if (const int i8 = atoi(ini("int8_swin", "-1").c_str()); i8 >= 0) SetEnvironmentVariableA("NR_INT8_SWIN", i8 ? "1" : "0");   // nr_runtime reads it at nr_create
    std::string lp = ini("log", "Rdna2NrRuntime.log"); if (!lp.empty()) { if (lp.size() < 2 || lp[1] != ':') lp = g_dir + lp; fopen_s(&g_log, lp.c_str(), "a"); }
    logf("Rdna2NrRuntime %s %s: data %s, model %s, model_res %s, history %d, nr_split %d, encode %d", __DATE__, __TIME__, s.dataDir.c_str(), s.model.c_str(), s.modelRes.c_str(), s.hist, s.split, s.enc);
}
BOOL WINAPI DllMain(HINSTANCE h, DWORD why, LPVOID) {
    if (why == DLL_PROCESS_ATTACH) { char p[MAX_PATH]; GetModuleFileNameA(h, p, MAX_PATH); g_dir = p; g_dir = g_dir.substr(0, g_dir.find_last_of('\\') + 1); }
    return TRUE;
}
static bool exists(const std::string& p) { return GetFileAttributesA(p.c_str()) != INVALID_FILE_ATTRIBUTES; }
static std::string narrow(const wchar_t* w) { if (!w) return ""; int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, 0, 0, 0, 0); std::string s(n ? n - 1 : 0, 0); if (n) WideCharToMultiByte(CP_UTF8, 0, w, -1, &s[0], n, 0, 0); return s; }

// ---------------------------------------------------------------- nr_runtime.dll (beside this DLL)
struct nr_ctx;
static struct {
    nr_ctx* (*create)(const char*, int, int, int, char*, int); void (*destroy)(nr_ctx*); void (*select)(const void*, int);
    int (*check)(const void*, int, char*, int); void (*weights)(const char*); void (*setlog)(void (*)(const char*));
    int (*cond)(nr_ctx*, int, float, float, float, int); int (*extOpen)(nr_ctx*, void*, unsigned long long, unsigned long long, unsigned long long, unsigned long long, void*, unsigned long long, unsigned long long, char*, int);
    int (*extHist)(nr_ctx*, unsigned long long, unsigned long long); int (*histLayout)(nr_ctx*, unsigned*, unsigned*);
    int (*enq2)(nr_ctx*, unsigned, int, int); int (*enq3)(nr_ctx*, unsigned, int, int, int); int (*splitn)(nr_ctx*, int, int*, int*, float*);
    int (*status)(nr_ctx*, unsigned*, unsigned*, int*); int (*chain)(nr_ctx*, nr_ctx*);
} nr; static bool g_nrOk = false;
static bool loadNr() {
    static std::mutex mu; std::lock_guard<std::mutex> g(mu); if (g_nrOk) return true;
    HMODULE m = LoadLibraryExA((g_dir + "nr_runtime.dll").c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!m) { fail(LMXXF_NR_UNAVAILABLE, "cannot load nr_runtime.dll beside Rdna2NrRuntime.dll (error %lu; HIP runtime amdhip64_6.dll from the AMD driver needed)", GetLastError()); return false; }
    auto P = [&](auto& f, const char* n) { f = (std::remove_reference_t<decltype(f)>)GetProcAddress(m, n); return f != nullptr; };
    bool ok = P(nr.create, "nr_create") & P(nr.destroy, "nr_destroy") & P(nr.select, "nr_select_device") & P(nr.check, "nr_check_device") & P(nr.weights, "nr_set_weights_dll") &
              P(nr.setlog, "nr_set_log") & P(nr.cond, "nr_set_conditioning") & P(nr.extOpen, "nr_ext_open") & P(nr.extHist, "nr_ext_history") & P(nr.histLayout, "nr_hist_layout") &
              P(nr.enq2, "nr_ext_enqueue2") & P(nr.enq3, "nr_ext_enqueue3") & P(nr.splitn, "nr_ext_splitn") & P(nr.status, "nr_ext_status") & P(nr.chain, "nr_ext_chain");
    if (!ok) { fail(LMXXF_NR_UNAVAILABLE, "nr_runtime.dll is too old (needs nr_check_device / nr_ext_status / nr_ext_chain)"); return false; }
    nr.setlog([](const char* s) { logf("%s", s); });
    return g_nrOk = true;
}

// ---------------------------------------------------------------- shaders (one source; colour / motion SRVs, out / base UAVs, io / st root UAVs)
static const char* kHlsl =
"Texture2D<float4> col : register(t0);\n"
"Texture2D<float4> mot : register(t1);\n"
"RWTexture2D<float4> outp : register(u0);\n"
"RWTexture2D<float4> base : register(u1);\n"
"globallycoherent RWByteAddressBuffer io : register(u2);\n"
"globallycoherent RWByteAddressBuffer st : register(u3);\n"
"cbuffer C : register(b0) { uint W; uint H; uint CW; uint CH; uint seq; uint maxIt; uint outOff; uint flagOff; uint histOff; uint histOut; uint hPWH; uint histOk;\n"
"  uint hMv; float mvSx; float mvSy; uint mWH; float hLo; float hHi; float tmap; uint enc; uint okMode; float ts; float cs; uint dbg; uint thresh; };\n"
"float san(float v) { return (v != v || v < -65504.0) ? 0.0 : min(v, 65504.0); }\n"
"float4 sn(float4 v) { return float4(san(v.x), san(v.y), san(v.z), san(v.w)); }\n"
"uint pk(float a, float b) { return f32tof16(san(a)) | (f32tof16(san(b)) << 16); }\n"
"float4 h4(uint2 q) { return float4(f16tof32(q.x & 0xFFFF), f16tof32(q.x >> 16), f16tof32(q.y & 0xFFFF), f16tof32(q.y >> 16)); }\n"
"float4 ldi(uint off, uint2 p) { return h4(io.Load2(off + (p.y * W + p.x) * 8)); }\n"
"float4 lds(uint2 p) { return h4(st.Load2(256 + (p.y * W + p.x) * 8)); }\n"
"float4 bili(float2 p, uint off) { p = clamp(p - 0.5, 0.0, float2(W - 1, H - 1)); uint2 a = (uint2)p; uint2 b = min(a + 1, uint2(W - 1, H - 1)); float2 t = p - a;\n"
"  return lerp(lerp(ldi(off, a), ldi(off, uint2(b.x, a.y)), t.x), lerp(ldi(off, uint2(a.x, b.y)), ldi(off, b), t.x), t.y); }\n"
"float4 bils(float2 p) { p = clamp(p - 0.5, 0.0, float2(W - 1, H - 1)); uint2 a = (uint2)p; uint2 b = min(a + 1, uint2(W - 1, H - 1)); float2 t = p - a;\n"
"  return lerp(lerp(lds(a), lds(uint2(b.x, a.y)), t.x), lerp(lds(uint2(a.x, b.y)), lds(b), t.x), t.y); }\n"
// encode: ldr = clamp to [0,1] (the port's LDR input); hdr = sRGB(x / (1 + x)) (the port's hdr_mode); auto = hdr once one frame had >= thresh pixels above 1 (sticky)
"bool hd() { return enc == 1 || (enc == 2 && st.Load(16) != 0); }\n"
"float3 T(float3 x) { x = max(x, 0.0); x = x / (1.0 + x); return lerp(1.055 * pow(x, 1.0 / 2.4) - 0.055, x * 12.92, step(x, 0.0031308)); }\n"
"float3 Ti(float3 y) { y = saturate(y); float3 z = lerp(pow((y + 0.055) / 1.055, 2.4), y / 12.92, step(y, 0.04045)); z = min(z, 0.999); return z / (1.0 - z); }\n"
"float4 E(float4 c) { c = sn(c); return float4(hd() ? T(c.rgb) : saturate(c.rgb), c.a); }\n"
"[numthreads(8,8,1)] void cpy(uint3 id : SV_DispatchThreadID) { if (id.x < CW && id.y < CH) base[id.xy] = sn(col[id.xy]); }\n"
"[numthreads(1,1,1)] void lat() { if (enc == 2) { if (st.Load(20) >= thresh) st.Store(16, 1u); st.Store(20, 0u); } }\n"
"[numthreads(8,8,1)] void imp(uint3 id : SV_DispatchThreadID) {\n"
"  if (id.x >= W || id.y >= H) return; float4 c;\n"
"  if (CW == W && CH == H) c = E(col[id.xy]);\n"
"  else { float2 p = (id.xy + 0.5) * float2(CW, CH) / float2(W, H), s = float2(CW, CH) / float2(W, H) * 0.25; c = 0;\n"   // 4 bilinear taps over the model pixel's footprint
"    for (int j = 0; j < 4; j++) { float2 q = clamp(p + s * float2((j & 1) ? 1 : -1, (j & 2) ? 1 : -1) - 0.5, 0.0, float2(CW - 1, CH - 1)); uint2 a = (uint2)q; uint2 b = min(a + 1, uint2(CW - 1, CH - 1)); float2 t = q - a;\n"
"      c += lerp(lerp(E(col[a]), E(col[uint2(b.x, a.y)]), t.x), lerp(E(col[uint2(a.x, b.y)]), E(col[b]), t.x), t.y); } c *= 0.25; }\n"
"  if (enc == 2) { float3 r = sn(col[min((uint2)((id.xy + 0.5) * float2(CW, CH) / float2(W, H)), uint2(CW - 1, CH - 1))]).rgb; if (max(r.r, max(r.g, r.b)) > 1.0) st.InterlockedAdd(20, 1); }\n"
"  io.Store2((id.y * W + id.x) * 8, uint2(pk(c.r, c.g), pk(c.b, c.a))); }\n"
// history slot (PAD_W x PAD_H RGB32F): the runtime's previous P (io+histOut) warped by this frame's vectors, colour-gated; current P where invalid
"float3 Pf(float3 x) { x = max(x, 0.0); if (tmap == 0.0) return x; x = max(tmap * x, 0.0); x = min(x / (1.0 + x), 1.0); return lerp(1.055 * pow(x, 1.0 / 2.4) - 0.055, x * 12.92, step(x, 0.0031308)); }\n"
"[numthreads(8,8,1)] void hwp(uint3 id : SV_DispatchThreadID) {\n"
"  uint PW = hPWH & 0xFFFF, PH = hPWH >> 16; if (id.x >= PW || id.y >= PH) return; uint o = histOff + (id.y * PW + id.x) * 12;\n"
"  if (id.x >= W || id.y >= H) { io.Store3(o, histOk != 0 ? io.Load3(histOut + (id.y * PW + id.x) * 12) : uint3(0, 0, 0)); return; }\n"   // padding: the network's own previous padding, as nr_run2's history

"  float3 cur = Pf(ldi(0, id.xy).rgb), res = cur;\n"
"  if (histOk != 0) { float2 uv = (id.xy + 0.5) / float2(W, H), m = 0;\n"
"    if (hMv) { uint2 mn = uint2(mWH & 0xFFFF, mWH >> 16); m = mot[min((uint2)(uv * mn), mn - 1)].xy * float2(mvSx, mvSy) / float2(CW, CH); }\n"   // colour px -> uv
"    float2 q = uv + m;\n"
"    if (all(m == m) && all(abs(m) < 0.25) && all(q >= 0.0) && all(q <= 1.0)) {\n"
"      float2 p = clamp(q * float2(W, H) - 0.5, 0.0, float2(W - 1, H - 1)); uint2 a0 = (uint2)p, b0 = min(a0 + 1, uint2(W - 1, H - 1)); float2 t = p - a0;\n"
"      float3 v00 = asfloat(io.Load3(histOut + (a0.y * PW + a0.x) * 12)), v10 = asfloat(io.Load3(histOut + (a0.y * PW + b0.x) * 12)), v01 = asfloat(io.Load3(histOut + (b0.y * PW + a0.x) * 12)), v11 = asfloat(io.Load3(histOut + (b0.y * PW + b0.x) * 12));\n"
"      float3 h = lerp(lerp(v00, v10, t.x), lerp(v01, v11, t.x), t.y), a = abs(h - cur);\n"
"      res = h + smoothstep(hLo, hHi, max(a.r, max(a.g, a.b))) * (cur - h); } }\n"   // = lerp(cur, h, w), exactly h at w = 1

"  io.Store3(o, asuint(res)); }\n"
"[numthreads(1,1,1)] void spin() { uint v = 0; for (uint i = 0; i < maxIt; i++) { io.InterlockedOr(flagOff, 0, v); if ((int)(v - seq) >= 0) break; } st.Store(0, (int)(v - seq) >= 0 ? 1u : 0u); }\n"
"[numthreads(8,8,1)] void cap(uint3 id : SV_DispatchThreadID) { if (id.x >= W || id.y >= H) return; float4 d = sn(ldi(outOff, id.xy)) - ldi(0, id.xy); st.Store2(256 + (id.y * W + id.x) * 8, uint2(pk(d.r, d.g), pk(d.b, 0))); }\n"
// okMode: 0 = st[0] from spin, 1 = finished (CPU saw it), 2 = passthrough, 3 = the captured edit (nr_split)
"[numthreads(8,8,1)] void exp(uint3 id : SV_DispatchThreadID) {\n"
"  if (id.x >= CW || id.y >= CH) return; float4 c = base[id.xy];\n"
"  if (okMode == 2 || (okMode == 0 && st.Load(0) == 0)) { outp[id.xy] = c; return; }\n"
"  float3 n, d; float2 p = (id.xy + 0.5) * float2(W, H) / float2(CW, CH);\n"
"  if (okMode == 3) { n = (CW == W && CH == H) ? E(c).rgb : bili(p, 0).rgb; d = (CW == W && CH == H) ? lds(id.xy).rgb : bils(p).rgb; }\n"
"  else if (CW == W && CH == H) { n = ldi(0, id.xy).rgb; d = sn(ldi(outOff, id.xy)).rgb - n; }\n"
"  else { n = bili(p, 0).rgb; d = sn(bili(p, outOff)).rgb - n; }\n"
"  if (ts != 1.0 || cs != 1.0) { float l = dot(d, float3(0.2126, 0.7152, 0.0722)); d = ts * l + cs * (d - l); }\n"   // Detail = luma of the edit, Colour = the rest
"  float3 o = hd() ? min(Ti(T(c.rgb) + d), 2.0 * max(c.rgb, 0.0) + 0.1) : max(c.rgb + d, 0.0);\n"   // LDR: frame + edit (excess above 1 survives); HDR: at most 2x + 0.1
"  if (dbg == 1) o = hd() ? Ti(n) : n; else if (dbg == 2) o = hd() ? Ti(n + d) : max(n + d, 0.0); else if (dbg == 3) o = abs(d) * 20.0; else if (dbg == 4) o += float3(0.25, 0, 0) * saturate(length(d) * 50.0);\n"
"  outp[id.xy] = float4(o, c.a); }\n";
enum { CB_W, CB_H, CB_CW, CB_CH, CB_SEQ, CB_MAXIT, CB_OUTOFF, CB_FLAGOFF, CB_HISTOFF, CB_HISTOUT, CB_PWH, CB_HISTOK, CB_HMV, CB_MVSX, CB_MVSY, CB_MWH, CB_HLO, CB_HHI, CB_TMAP, CB_ENC, CB_OK, CB_TS, CB_CS, CB_DBG, CB_THRESH, CB_N };
enum { P_CPY, P_LAT, P_IMP, P_HWP, P_SPIN, P_CAP, P_EXP, P_N };
static const char* kEntry[P_N] = {"cpy", "lat", "imp", "hwp", "spin", "cap", "exp"};
static const int kRing = 64;   // descriptor sets (colour, motion, out, base) in flight

// ---------------------------------------------------------------- context
template <class T> static void rel(T*& p) { if (p) p->Release(); p = nullptr; }
struct Net {   // one network size (+ its D3D12 buffers); replaced as a whole when the size / model / passes change
    std::string key, dir; unsigned W = 0, H = 0, CW = 0, CH = 0; int passes = 1, ve = 1, split = 0;
    ID3D12Resource *io = nullptr, *sig = nullptr, *st = nullptr, *out = nullptr, *base = nullptr; HANDLE hIo = nullptr, hSig = nullptr;
    UINT64 ioAlloc = 0, ioSize = 0, sigAlloc = 0, outOff = 0, flagOff = 0, histOff = 0, histOut = 0; unsigned padW = 0, padH = 0; float impScale = 0; bool histOn = false;
    nr_ctx* nc[3] = {}; std::atomic<int> state{0}; std::string err; std::thread th; unsigned buried = 0;   // state 0 loading, 1 ready, 2 failed
    float cond[5] = {NAN, NAN, NAN, NAN, NAN};   // conditioning last applied
    int hipPart = 0; unsigned hipLast = 0; bool editValid = false;   // nr_split: parts released for the HIP job in flight, its last part's seq, a captured edit exists
    void freeHip() {   // nr_destroy closes the ext thread first (it finishes the job it runs), so the HIP side never outlives its work
        if (th.joinable()) th.join();
        for (auto& x : nc) { if (x) nr.destroy(x); x = nullptr; }   // the first one owns the ext thread and the chain
    }
    ~Net() {
        freeHip();
        rel(io); rel(sig); rel(st); rel(out); rel(base); if (hIo) CloseHandle(hIo); if (hSig) CloseHandle(hSig);
    }
};
struct Job { uint32_t state = LMXXF_NR_JOB_NONE; unsigned seq = 0; Net* net = nullptr; bool pass = true, launched = false, async = true, hist = false; int part = 0; float ts = 1, cs = 1; uint32_t dbg = 0;
             ID3D12Resource* color = nullptr; uint32_t colorState = 0; ID3D12Resource* motion = nullptr; uint32_t motionState = 0; unsigned mw = 0, mh = 0; float msx = 0, msy = 0; };
struct Ctx {
    std::mutex mu; ID3D12Device* dev = nullptr; ID3D12CommandQueue* queue = nullptr; LUID luid{};
    std::string dataBase, modelId, weights; ID3D12RootSignature* rs = nullptr; ID3D12PipelineState* pso[P_N] = {}; ID3D12DescriptorHeap* heap = nullptr; UINT incr = 0; unsigned ring = 0;
    Net* net = nullptr; std::vector<Net*> grave; Job jobs[8]; unsigned seq = 0; bool histReset = true, lastLaunched = false, vkBridge = false; uint64_t frames = 0, nrJobs = 0, passJobs = 0;
    std::string lastNote;
};

// ---------------------------------------------------------------- helpers
static ID3D12Resource* buffer(ID3D12Device* d, UINT64 size, D3D12_HEAP_FLAGS hf, UINT64* alloc) {
    D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd = {}; rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; rd.Width = size; rd.Height = 1; rd.DepthOrArraySize = 1;
    rd.MipLevels = 1; rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ID3D12Resource* r = nullptr; if (FAILED(d->CreateCommittedResource(&hp, hf, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&r)))) return nullptr;
    if (alloc) *alloc = d->GetResourceAllocationInfo(0, 1, &rd).SizeInBytes; return r;
}
static ID3D12Resource* texture(ID3D12Device* d, unsigned w, unsigned h, D3D12_RESOURCE_STATES st) {
    D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_DEFAULT; D3D12_RESOURCE_DESC rd = {}; rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; rd.Width = w; rd.Height = h;
    rd.DepthOrArraySize = 1; rd.MipLevels = 1; rd.Format = DXGI_FORMAT_R16G16B16A16_FLOAT; rd.SampleDesc.Count = 1; rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ID3D12Resource* r = nullptr; return SUCCEEDED(d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, st, nullptr, IID_PPV_ARGS(&r))) ? r : nullptr;
}
static DXGI_FORMAT srvFormat(DXGI_FORMAT f) {
    switch (f) { case DXGI_FORMAT_R16G16B16A16_TYPELESS: return DXGI_FORMAT_R16G16B16A16_FLOAT; case DXGI_FORMAT_R32G32B32A32_TYPELESS: return DXGI_FORMAT_R32G32B32A32_FLOAT;
                 case DXGI_FORMAT_R8G8B8A8_TYPELESS: return DXGI_FORMAT_R8G8B8A8_UNORM; case DXGI_FORMAT_R10G10B10A2_TYPELESS: return DXGI_FORMAT_R10G10B10A2_UNORM;
                 case DXGI_FORMAT_R16G16_TYPELESS: return DXGI_FORMAT_R16G16_FLOAT; case DXGI_FORMAT_R32G32_TYPELESS: return DXGI_FORMAT_R32G32_FLOAT;
                 case DXGI_FORMAT_B8G8R8A8_TYPELESS: return DXGI_FORMAT_B8G8R8A8_UNORM; default: return f; }
}
static void barrier(ID3D12GraphicsCommandList* cl, ID3D12Resource* r, UINT a, UINT b) {
    if (a == b) return; D3D12_RESOURCE_BARRIER x = {}; x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; x.Transition.pResource = r;
    x.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES; x.Transition.StateBefore = (D3D12_RESOURCE_STATES)a; x.Transition.StateAfter = (D3D12_RESOURCE_STATES)b; cl->ResourceBarrier(1, &x);
}
static bool seqReached(unsigned v, unsigned s) { return (int)(v - s) >= 0; }
static const UINT kSrv = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, kUav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

static bool readMeta(const std::string& dir, unsigned* w, unsigned* h, int* ve) {
    FILE* f = nullptr; fopen_s(&f, (dir + "\\meta.txt").c_str(), "r"); if (!f) return false; char l[256]; *w = *h = 0; *ve = 1;
    while (fgets(l, sizeof l, f)) { unsigned v; if (sscanf_s(l, "width=%u", &v) == 1) *w = v; if (sscanf_s(l, "height=%u", &v) == 1) *h = v; if (sscanf_s(l, "vit_every=%u", &v) == 1) *ve = (int)v; }
    fclose(f); return *w && *h;
}
static float importScale(const std::string& dir) {   // k_import kernarg +0x28 mode, +0x2c scale (0 = mode 0, raw colour) - as dll/dlssnr_nr.cpp
    FILE* f = nullptr; fopen_s(&f, (dir + "\\manifest.txt").c_str(), "r"); if (!f) return 0; char l[1024]; size_t off = 0, len = 0; bool hit = false;
    while (!hit && fgets(l, sizeof l, f)) { std::string t(l); size_t a = t.find('|'); if (a == std::string::npos || t.substr(a + 1, t.find('|', a + 1) - a - 1).find("k_import") == std::string::npos) continue;
        size_t b = t.find('|', a + 1), c = t.find('|', b + 1); off = strtoull(t.c_str() + b + 1, nullptr, 10); len = strtoull(t.c_str() + c + 1, nullptr, 10); hit = len >= 0x30; }
    fclose(f); if (!hit) return 0;
    fopen_s(&f, (dir + "\\kernargs.bin").c_str(), "rb"); if (!f) return 0; int mode = 0; float sc = 0; _fseeki64(f, (long long)off + 0x28, SEEK_SET);
    bool ok = fread(&mode, 4, 1, f) == 1 && fread(&sc, 4, 1, f) == 1; fclose(f); return ok && mode != 0 && sc > 0 && sc < 1e6f ? sc : 0;
}
// Step for a colour size: model_res if it exists, else the step nearest in height to colour height x model_scale (ties: the smaller one).
static std::string pickStep(const std::string& base, unsigned w, unsigned h, float scale) {
    std::vector<std::pair<unsigned, unsigned>> st; WIN32_FIND_DATAA fd; HANDLE fh = FindFirstFileA((base + "\\*x*").c_str(), &fd);
    if (fh != INVALID_HANDLE_VALUE) { do { unsigned a, b; if ((fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && sscanf_s(fd.cFileName, "%ux%u", &a, &b) == 2 && exists(base + "\\" + fd.cFileName + "\\meta.txt")) st.push_back({a, b}); } while (FindNextFileA(fh, &fd)); FindClose(fh); }
    if (st.empty()) return "";
    char n[32];
    for (auto& p : st) { snprintf(n, sizeof n, "%ux%u", p.first, p.second); if (!_stricmp(n, g_set.modelRes.c_str())) return base + "\\" + n; }
    std::sort(st.begin(), st.end(), [](auto& a, auto& b) { return a.second < b.second; });
    const float want = h * (scale > 0.f ? scale : 1.f); auto best = st[0];
    for (auto& p : st) if (fabsf(p.second - want) < fabsf(best.second - want)) best = p;
    snprintf(n, sizeof n, "%ux%u", best.first, best.second); return base + "\\" + n;
}
// RDNA2 discrete GPU by PCI id (Navi 21/22/23: 0x73A0-0x73FF, Navi 24: 0x7420-0x743F); RDNA2 APUs (Rembrandt, Raphael, Van Gogh, Mendocino) have other ids.
static bool rdna2Dgpu(UINT ven, UINT id) { return ven == 0x1002 && ((id >= 0x73A0 && id <= 0x73FF) || (id >= 0x7420 && id <= 0x743F)); }

// ---------------------------------------------------------------- network build (D3D12 side on the API thread, HIP side on a loader thread)
static void applyCond(Net* n, const float c[5]) {   // c: tone, structure, skin, other, style (lmxxf controls; NaN = built-in) -> nr_set_conditioning
    if (!memcmp(n->cond, c, sizeof n->cond)) return; memcpy(n->cond, c, sizeof n->cond);
    float tone = 1, structure = 1, skin = -1; int mask = 1, style = 0;
    if (c[0] == c[0]) {   // lmxxf's pre-block features: mask on = (tone, 1, skin, structure), mask off = (tone, structure, -1, -1); style: 1 = shipped
        tone = std::clamp(c[0], 0.f, 2.f); mask = !(c[2] <= -1.f && c[3] <= -1.f);
        if (mask) { skin = std::clamp(c[2], -1.f, 2.f); structure = std::clamp(c[3], 0.f, 2.f); } else structure = std::clamp(c[1], 0.f, 2.f);
        style = c[4] == 1.f ? 0 : std::clamp((int)lrintf(c[4]), 0, 2);
    }
    for (nr_ctx* x : n->nc) if (x) nr.cond(x, style, tone, structure, skin, mask);
    logf("conditioning: style %d tone %.2f structure %.2f skin %.2f mask %d", style, tone, structure, skin, mask);
}
static void loader(Ctx* C, Net* n) {
    char err[512] = "";
    auto bad = [&](const char* what) { n->err = std::string(what) + ": " + err; logf("network %s: %s", n->key.c_str(), n->err.c_str()); n->state = 2; };
    nr.select(&C->luid, g_set.allowAny); nr.weights(C->weights.c_str());
    const int ve = n->ve > 1 && g_set.reuseBn ? -n->ve : n->ve; const DWORD t0 = GetTickCount();
    for (int i = 0; i < n->passes; i++) if (!(n->nc[i] = nr.create(n->dir.c_str(), (int)n->W, (int)n->H, ve, err, sizeof err))) return bad("nr_create");
    if (nr.extOpen(n->nc[0], n->hIo, n->ioAlloc, n->ioSize, n->outOff, n->flagOff, n->hSig, n->sigAlloc, 65536, err, sizeof err)) return bad("nr_ext_open");
    for (int i = 1; i < n->passes; i++) if (nr.chain(n->nc[i - 1], n->nc[i])) { snprintf(err, sizeof err, "pass %d", i + 1); return bad("nr_ext_chain"); }
    if (g_set.hist) { unsigned pw = 0, ph = 0; if (!nr.histLayout(n->nc[0], &pw, &ph) && pw >= n->W && ph >= n->H && (UINT64)pw * ph * 12 <= n->histOut - n->histOff && !nr.extHist(n->nc[0], n->histOff, n->histOut)) { n->padW = pw; n->padH = ph; n->histOn = true; }
                      else logf("history: unavailable for %s", n->dir.c_str()); }
    if (n->split) { int kf[8] = {}, ks[8] = {}; float ms[16] = {}; if (nr.splitn(n->nc[0], n->split, kf, ks, ms)) { logf("nr_split: setup failed - whole network per job"); n->split = 0; } else logf("nr_split: %d parts", n->split); }
    float c[5] = {NAN, NAN, NAN, NAN, NAN}; n->cond[0] = 0; applyCond(n, c);   // NVIDIA's defaults until the host sends controls
    logf("network ready: %s, %ux%u, colour %ux%u, passes %d, vit_every %d, history %s, %lu ms", n->dir.c_str(), n->W, n->H, n->CW, n->CH, n->passes, n->ve, n->histOn ? "on" : "off", GetTickCount() - t0);
    n->state = 1;
}
static Net* makeNet(Ctx* C, const std::string& dir, unsigned W, unsigned H, unsigned cw, unsigned ch, int passes, int ve) {
    Net* n = new Net; n->dir = dir; n->W = W; n->H = H; n->CW = cw; n->CH = ch; n->passes = passes; n->ve = ve; n->split = g_set.split; n->impScale = importScale(dir);
    const UINT64 b = (UINT64)W * H * 8, a = (b + 65535) & ~65535ull, hb = ((UINT64)((W + 255) & ~127u) * ((H + 255) & ~127u) * 12 + 65535) & ~65535ull;   // history: room for the data dir's PAD_W x PAD_H (e.g. 1344x768 at 1280x720)
    n->outOff = a; n->flagOff = 2 * a; n->histOff = 2 * a + 65536; n->histOut = n->histOff + hb; n->ioSize = n->histOut + hb;
    ID3D12Device* d = C->dev;
    n->io = buffer(d, n->ioSize, D3D12_HEAP_FLAG_SHARED, &n->ioAlloc); n->sig = buffer(d, 65536, D3D12_HEAP_FLAG_SHARED, &n->sigAlloc); n->st = buffer(d, 256 + b, D3D12_HEAP_FLAG_NONE, nullptr);
    n->out = texture(d, cw, ch, (D3D12_RESOURCE_STATES)kSrv); n->base = texture(d, cw, ch, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (!n->io || !n->sig || !n->st || !n->out || !n->base || FAILED(d->CreateSharedHandle(n->io, nullptr, GENERIC_ALL, nullptr, &n->hIo)) || FAILED(d->CreateSharedHandle(n->sig, nullptr, GENERIC_ALL, nullptr, &n->hSig))) {
        n->err = "D3D12 shared buffers"; n->state = 2; return n; }
    n->th = std::thread(loader, C, n); g_imports += 2;
    return n;
}
static void bury(Ctx* C, Net* n) {   // HIP side freed now (VRAM back at once), D3D12 buffers kept: lists the host recorded may still read them
    if (!n) return; if (n->state != 0) n->freeHip(); n->buried = 0; C->grave.push_back(n);
}
static void sweep(Ctx* C, bool all) {   // ponytail: a buried network is freed 240 PrepareFrames later (the host's older lists are long done); a fence per list if that is ever too early
    for (auto it = C->grave.begin(); it != C->grave.end();) { if (all || ++(*it)->buried > 240) { delete *it; it = C->grave.erase(it); } else ++it; }
}

// ---------------------------------------------------------------- recording
static D3D12_GPU_DESCRIPTOR_HANDLE descriptors(Ctx* C, Net* n, Job* j) {   // colour, motion SRVs (null when absent), out, base UAVs
    const UINT k = (C->ring++ % kRing) * 4; auto cpu = C->heap->GetCPUDescriptorHandleForHeapStart(); auto gpu = C->heap->GetGPUDescriptorHandleForHeapStart();
    auto at = [&](UINT i) { D3D12_CPU_DESCRIPTOR_HANDLE c = cpu; c.ptr += (SIZE_T)(k + i) * C->incr; return c; }; gpu.ptr += (UINT64)k * C->incr;
    D3D12_SHADER_RESOURCE_VIEW_DESC sv = {}; sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D; sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; sv.Texture2D.MipLevels = 1;
    sv.Format = j && j->color ? srvFormat(j->color->GetDesc().Format) : DXGI_FORMAT_R16G16B16A16_FLOAT; C->dev->CreateShaderResourceView(j ? j->color : nullptr, &sv, at(0));
    sv.Format = j && j->motion ? srvFormat(j->motion->GetDesc().Format) : DXGI_FORMAT_R16G16_FLOAT; C->dev->CreateShaderResourceView(j ? j->motion : nullptr, &sv, at(1));
    D3D12_UNORDERED_ACCESS_VIEW_DESC uv = {}; uv.Format = DXGI_FORMAT_R16G16B16A16_FLOAT; uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    C->dev->CreateUnorderedAccessView(n->out, nullptr, &uv, at(2)); C->dev->CreateUnorderedAccessView(n->base, nullptr, &uv, at(3));
    return gpu;
}
static void constants(Net* n, Job* j, UINT* cb) {
    memset(cb, 0, CB_N * 4); auto F = [&](int i, float v) { memcpy(&cb[i], &v, 4); };
    cb[CB_W] = n->W; cb[CB_H] = n->H; cb[CB_CW] = n->CW; cb[CB_CH] = n->CH; cb[CB_SEQ] = j->seq; cb[CB_MAXIT] = (UINT)g_set.spinMs * 6000u;   // ~0.16 us per poll on a 6900 XT
    cb[CB_OUTOFF] = (UINT)n->outOff; cb[CB_FLAGOFF] = (UINT)n->flagOff; cb[CB_HISTOFF] = (UINT)n->histOff; cb[CB_HISTOUT] = (UINT)n->histOut; cb[CB_PWH] = n->padW | (n->padH << 16);
    cb[CB_HISTOK] = j->hist; cb[CB_HMV] = j->motion != nullptr; F(CB_MVSX, j->msx); F(CB_MVSY, j->msy); cb[CB_MWH] = j->mw | (j->mh << 16);
    F(CB_HLO, g_set.hLo); F(CB_HHI, g_set.hHi); F(CB_TMAP, n->impScale); cb[CB_ENC] = g_set.enc; F(CB_TS, j->ts); F(CB_CS, j->cs); cb[CB_DBG] = j->dbg;
    cb[CB_THRESH] = (UINT)(g_set.hdrFrac * n->W * n->H);
}
static void bind(Ctx* C, Net* n, ID3D12GraphicsCommandList* cl, D3D12_GPU_DESCRIPTOR_HANDLE t, const UINT* cb) {
    ID3D12DescriptorHeap* hp[] = {C->heap}; cl->SetDescriptorHeaps(1, hp); cl->SetComputeRootSignature(C->rs); cl->SetComputeRootDescriptorTable(0, t);
    cl->SetComputeRootUnorderedAccessView(1, n->io->GetGPUVirtualAddress()); cl->SetComputeRootUnorderedAccessView(2, n->st->GetGPUVirtualAddress()); cl->SetComputeRoot32BitConstants(3, CB_N, cb, 0);
}
static void run(Ctx* C, ID3D12GraphicsCommandList* cl, int p, unsigned gx, unsigned gy) {
    D3D12_RESOURCE_BARRIER u = {}; u.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; cl->SetPipelineState(C->pso[p]); cl->Dispatch(gx, gy, 1); cl->ResourceBarrier(1, &u);
}

// ---------------------------------------------------------------- API
#define CTX(c) Ctx* C = (Ctx*)(c); if (!C) return fail(LMXXF_NR_INVALID_ARGUMENT, "null context"); std::lock_guard<std::mutex> lock_(C->mu)
#define JOB(j) Job* J = (Job*)(j); if (!J || J < C->jobs || J >= C->jobs + 8) return fail(LMXXF_NR_INVALID_ARGUMENT, "not a job of this context")
static int32_t QueryCapabilities(LmxxfNrCapabilities* o) {
    clearErr(); if (!o || o->struct_size < sizeof(LmxxfNrCapabilities)) return fail(LMXXF_NR_INVALID_ARGUMENT, "QueryCapabilities: struct_size");
    o->abi_version = LMXXF_NR_ABI_VERSION; o->max_input_width = 3840; o->max_input_height = 2160; o->history_supported = 1; o->overlap_supported = 1; o->graph_supported = 0;
    o->hip_ready = g_nrOk; o->gfx1201_target = 0; return LMXXF_NR_OK;
}
static int32_t Create(const LmxxfNrCreateInfo* ci, void** out) {
    clearErr(); std::call_once(g_setOnce, loadSettings);
    if (!ci || !out || ci->struct_size < sizeof(LmxxfNrCreateInfo) || !ci->device || !ci->queue) return fail(LMXXF_NR_INVALID_ARGUMENT, "Create: struct_size / device / queue");
    if (ci->flags & ~(LMXXF_NR_CREATE_FLAG_ZERO_OUTPUT_FALLBACK | LMXXF_NR_CREATE_FLAG_ACO)) logf("Create: unknown flags 0x%x ignored", ci->flags);   // ACO is mochizuki-only; outputs start zeroed here anyway
    *out = nullptr; ID3D12Device* dev = (ID3D12Device*)ci->device; ID3D12CommandQueue* q = (ID3D12CommandQueue*)ci->queue;
    { ID3D12Device* qd = nullptr; if (FAILED(q->GetDevice(IID_PPV_ARGS(&qd))) || !qd) return fail(LMXXF_NR_INVALID_ARGUMENT, "Create: queue has no device");
      const bool same = qd == dev; qd->Release(); if (!same) return fail(LMXXF_NR_INVALID_ARGUMENT, "Create: queue belongs to another device"); }
    const LUID luid = dev->GetAdapterLuid();
    { IDXGIFactory4* f = nullptr; IDXGIAdapter1* a = nullptr; DXGI_ADAPTER_DESC1 d = {};
      if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&f))) || FAILED(f->EnumAdapterByLuid(luid, IID_PPV_ARGS(&a))) || FAILED(a->GetDesc1(&d))) { rel(a); rel(f); return fail(LMXXF_NR_UNAVAILABLE, "Create: adapter of the device not found"); }
      rel(a); rel(f); char nm[128]; WideCharToMultiByte(CP_UTF8, 0, d.Description, -1, nm, sizeof nm, 0, 0);
      logf("Create: adapter %s (vendor %04x device %04x, %llu MB VRAM)", nm, d.VendorId, d.DeviceId, (unsigned long long)(d.DedicatedVideoMemory >> 20));
      if (!rdna2Dgpu(d.VendorId, d.DeviceId) && !g_set.allowAny) return fail(LMXXF_NR_UNAVAILABLE, "Create: %s (%04x:%04x) is not an RDNA2 discrete GPU (RX 6000); allow_any_gpu=1 overrides", nm, d.VendorId, d.DeviceId); }
    if (!loadNr()) return LMXXF_NR_UNAVAILABLE;
    { char e[512] = ""; if (nr.check(&luid, g_set.allowAny, e, sizeof e) < 0) return fail(LMXXF_NR_UNAVAILABLE, "Create: %s", e); }
    // data + weights: nr_data next to this DLL (ini data_dir) or in assets_directory; the user's nvngx_dlssnr.dll (310.8 / 310.9.16) for the weights
    std::string data = g_set.dataDir, assets = narrow(ci->assets_directory); if (!assets.empty() && assets.back() != '\\') assets += '\\';
    if (!exists(data) && !assets.empty() && exists(assets + "nr_data")) data = assets + "nr_data";
    char exe[MAX_PATH] = ""; GetModuleFileNameA(nullptr, exe, MAX_PATH); std::string ed = exe; ed = ed.substr(0, ed.find_last_of('\\') + 1);
    std::string w; for (const std::string& p : {g_set.weightsDll, g_dir + "nvngx_dlssnr.dll", data + "\\..\\nvngx_dlssnr.dll", assets.empty() ? std::string() : assets + "nvngx_dlssnr.dll", ed + "nvngx_dlssnr.dll"})
        if (!p.empty() && exists(p)) { w = p; break; }
    if (w.empty()) return fail(LMXXF_NR_UNAVAILABLE, "Create: nvngx_dlssnr.dll (NVIDIA's DLSS-NR model, 310.8 or 310.9.16) not found next to Rdna2NrRuntime.dll, the assets folder or the game");
    std::string id; char hx[17];
    auto have = [&](uint64_t v) { if (!v) return false; snprintf(hx, sizeof hx, "%016llx", (unsigned long long)v); return exists(data + "\\" + hx); };
    const bool wantOld = !_stricmp(g_set.model.c_str(), "old") || !_stricmp(g_set.model.c_str(), "large"), wantNew = !_stricmp(g_set.model.c_str(), "new") || !_stricmp(g_set.model.c_str(), "small");
    if (!wantOld && have(nr_model_id(w, "WEIGHTS_CC_SMALL_FP8"))) id = hx;
    else if (!wantNew && have(nr_model_id(w, "WEIGHTS_HT"))) id = hx;
    if (id.empty()) return fail(LMXXF_NR_UNAVAILABLE, "Create: %s has no model this runtime has data for in %s (model=%s)", w.c_str(), data.c_str(), g_set.model.c_str());
    // shaders, root signature, descriptor heap
    typedef HRESULT(WINAPI * PFN_Compile)(const void*, SIZE_T, const char*, const void*, void*, const char*, const char*, UINT, UINT, ID3DBlob**, ID3DBlob**);
    HMODULE dc = LoadLibraryExW(L"d3dcompiler_47.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32); auto comp = dc ? (PFN_Compile)GetProcAddress(dc, "D3DCompile") : nullptr;
    if (!comp) return fail(LMXXF_NR_UNAVAILABLE, "Create: d3dcompiler_47.dll missing");
    Ctx* C = new Ctx; C->dev = dev; C->queue = q; dev->AddRef(); q->AddRef(); C->luid = luid; C->dataBase = data + "\\" + id; C->modelId = id; C->weights = w;
    auto bail = [&](const char* m) { int r = fail(LMXXF_NR_FAILED, "Create: %s", m); for (auto& p : C->pso) rel(p); rel(C->rs); rel(C->heap); rel(C->queue); rel(C->dev); delete C; return r; };
    D3D12_DESCRIPTOR_RANGE rg[2] = {}; rg[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; rg[0].NumDescriptors = 2; rg[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV; rg[1].NumDescriptors = 2; rg[1].OffsetInDescriptorsFromTableStart = 2;
    D3D12_ROOT_PARAMETER rp[4] = {}; rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; rp[0].DescriptorTable.NumDescriptorRanges = 2; rp[0].DescriptorTable.pDescriptorRanges = rg;
    rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; rp[1].Descriptor.ShaderRegister = 2; rp[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; rp[2].Descriptor.ShaderRegister = 3;
    rp[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; rp[3].Constants.Num32BitValues = CB_N;
    D3D12_ROOT_SIGNATURE_DESC rsd = {}; rsd.NumParameters = 4; rsd.pParameters = rp; ID3DBlob *sb = nullptr, *er = nullptr;
    if (FAILED(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sb, &er)) || FAILED(dev->CreateRootSignature(0, sb->GetBufferPointer(), sb->GetBufferSize(), IID_PPV_ARGS(&C->rs)))) { rel(sb); rel(er); return bail("root signature"); }
    rel(sb); rel(er);
    for (int i = 0; i < P_N; i++) {
        ID3DBlob* cs = nullptr; er = nullptr;
        if (FAILED(comp(kHlsl, strlen(kHlsl), "rdna2_nr", nullptr, nullptr, kEntry[i], "cs_5_0", 1 << 15, 0, &cs, &er))) { logf("shader %s: %s", kEntry[i], er ? (char*)er->GetBufferPointer() : "?"); rel(er); return bail("shader compile"); }
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {}; pd.pRootSignature = C->rs; pd.CS = {cs->GetBufferPointer(), cs->GetBufferSize()};
        HRESULT h = dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&C->pso[i])); rel(cs); if (FAILED(h)) return bail("pipeline state");
    }
    D3D12_DESCRIPTOR_HEAP_DESC hd = {}; hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors = 4 * kRing; hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&C->heap)))) return bail("descriptor heap");
    C->incr = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    logf("Create: ok, model %s (%s), weights %s", id.c_str(), C->dataBase.c_str(), w.c_str());
    *out = C; return LMXXF_NR_OK;
}
static int32_t PrepareSession(void* c) { clearErr(); CTX(c); return LMXXF_NR_OK; }
static int waitDone(Ctx* C, Job* J, DWORD ms) {   // CPU wait until the ext thread finished with the job (ran or dropped); 0 = done
    if (!J->launched || !J->net || !J->net->nc[0]) return 0; const DWORD t0 = GetTickCount(); unsigned pr = 0;
    while (!nr.status(J->net->nc[0], nullptr, &pr, nullptr) && !seqReached(pr, J->seq)) { if (GetTickCount() - t0 > ms) return 1; Sleep(1); }
    return 0;
}
static int32_t Destroy(void* c) {
    clearErr(); Ctx* C = (Ctx*)c; if (!C) return LMXXF_NR_OK;
    { std::lock_guard<std::mutex> g(C->mu); for (auto& j : C->jobs) waitDone(C, &j, 2000); bury(C, C->net); C->net = nullptr; sweep(C, true);
      for (auto& p : C->pso) rel(p); rel(C->rs); rel(C->heap); rel(C->queue); rel(C->dev); logf("Destroy: %llu frames, %llu network jobs, %llu passthrough", C->frames, C->nrJobs, C->passJobs); }
    delete C; return LMXXF_NR_OK;
}
// OpenNR host (LmxxfNrApi.h of OpenNR): the 136-byte table, FrameInfo up to offsetof(depth) = 136, Job 24 (32 with FIRST_PASS).
// Used here: STRENGTH, DEBUG_VIEW, CODEC_PASSTHROUGH, TEMPORAL, FIRST_PASS; the other flags (smoothing, style, int4, ...) name
// features of other runtimes and are ignored (logged once).
static const uint32_t kUsedFlags = LMXXF_NR_FRAME_FLAG_STRENGTH | LMXXF_NR_FRAME_FLAG_DEBUG_VIEW | LMXXF_NR_FRAME_FLAG_CODEC_PASSTHROUGH |
                                   LMXXF_NR_FRAME_FLAG_TEMPORAL | LMXXF_NR_FRAME_FLAG_FIRST_PASS;
static int32_t PrepareFrame(void* c, const LmxxfNrFrameInfo* fi, LmxxfNrJob* job) {
    clearErr(); CTX(c);
    if (!fi || !job || job->struct_size < offsetof(LmxxfNrJob, first_pass_output)) return fail(LMXXF_NR_INVALID_ARGUMENT, "PrepareFrame: null / job struct_size");
    const uint32_t ss = fi->struct_size;
    if (ss < 64 || ss > sizeof(LmxxfNrFrameInfo) || (ss & 3)) return fail(LMXXF_NR_INVALID_ARGUMENT, "PrepareFrame: struct_size %u", ss);
    { static uint32_t seen = 0; const uint32_t extra = fi->flags & ~kUsedFlags & ~seen; if (extra) { seen |= extra; logf("PrepareFrame: flags 0x%x are not used by this runtime (ignored)", extra); } }
    if (!fi->color_width || !fi->color_height || fi->color_width > 8192 || fi->color_height > 8192) return fail(LMXXF_NR_INVALID_ARGUMENT, "PrepareFrame: colour size %ux%u", fi->color_width, fi->color_height);
    auto has = [&](size_t end) { return ss >= end; };   // fields beyond a smaller struct read as defaults
    const bool temporal = (fi->flags & LMXXF_NR_FRAME_FLAG_TEMPORAL) && has(offsetof(LmxxfNrFrameInfo, passes) + 4);
    const unsigned passes = has(offsetof(LmxxfNrFrameInfo, passes) + 4) ? std::clamp(fi->passes, 1u, 3u) : 1u;
    const float scale = has(offsetof(LmxxfNrFrameInfo, model_scale) + 4) && fi->model_scale >= 0.25f && fi->model_scale <= 1.f ? fi->model_scale : 1.f;
    if (fi->flags & LMXXF_NR_FRAME_FLAG_STRENGTH) { if (!(fi->transfer_strength >= 0.f && fi->transfer_strength <= 2.f && fi->color_strength >= 0.f && fi->color_strength <= 4.f)) return fail(LMXXF_NR_INVALID_ARGUMENT, "PrepareFrame: strength out of range"); }
    float ctl[5] = {NAN, NAN, NAN, NAN, NAN};   // OpenNR sends no pre-block controls: NVIDIA's defaults
    Job* J = nullptr; for (auto& j : C->jobs) if (j.state == LMXXF_NR_JOB_NONE || j.state == LMXXF_NR_JOB_RETIRED) { J = &j; break; }
    if (!J) return fail(LMXXF_NR_FAILED, "PrepareFrame: all 8 jobs are outstanding (retire or abandon them)");
    C->frames++; sweep(C, false);
    // network for this colour size / passes / schedule
    const std::string step = pickStep(C->dataBase, fi->color_width, fi->color_height, scale);
    if (step.empty()) return fail(LMXXF_NR_UNAVAILABLE, "PrepareFrame: no resolution step in %s", C->dataBase.c_str());
    unsigned W, H; int metaVe; if (!readMeta(step, &W, &H, &metaVe)) return fail(LMXXF_NR_UNAVAILABLE, "PrepareFrame: bad meta.txt in %s", step.c_str());
    const int ve = g_set.vitEvery ? g_set.vitEvery : metaVe;
    char key[1024]; snprintf(key, sizeof key, "%s|%ux%u|p%u|v%d", step.c_str(), fi->color_width, fi->color_height, passes, ve);
    if (!C->net || C->net->key != key) {
        for (auto& j : C->jobs) if (j.launched && j.state < LMXXF_NR_JOB_CONSUMER_COMPLETE && waitDone(C, &j, 2000)) return fail(LMXXF_NR_FAILED, "PrepareFrame: a job of the old network did not finish");
        for (auto& j : C->jobs) if (j.net == C->net && j.state != LMXXF_NR_JOB_NONE) j.state = LMXXF_NR_JOB_RETIRED;
        logf("network: %s -> %s", C->net ? C->net->key.c_str() : "(none)", key);
        bury(C, C->net); C->net = makeNet(C, step, W, H, fi->color_width, fi->color_height, (int)passes, ve); C->net->key = key; C->histReset = true;
    }
    Net* n = C->net;
    if (n->state == 2) return fail(LMXXF_NR_FAILED, "PrepareFrame: network failed to load: %s", n->err.c_str());
    *J = Job(); J->net = n; J->state = LMXXF_NR_JOB_PREPARED; J->color = (ID3D12Resource*)fi->color; J->colorState = fi->color_state;
    J->pass = n->state != 1 || (fi->flags & LMXXF_NR_FRAME_FLAG_CODEC_PASSTHROUGH);
    if (fi->flags & LMXXF_NR_FRAME_FLAG_STRENGTH) { J->ts = std::min(fi->transfer_strength, 1.f); J->cs = std::min(fi->color_strength, 1.f); }   // above 1 (extrapolation) is not supported here: capped
    if (fi->flags & LMXXF_NR_FRAME_FLAG_DEBUG_VIEW) J->dbg = std::min(fi->debug_view, 4u);
    if (temporal && fi->motion && fi->motion_width && fi->motion_height && fi->motion_scale_x == fi->motion_scale_x && fi->motion_scale_y == fi->motion_scale_y) {
        J->motion = (ID3D12Resource*)fi->motion; J->motionState = fi->motion_state; J->mw = fi->motion_width; J->mh = fi->motion_height; J->msx = fi->motion_scale_x; J->msy = fi->motion_scale_y; }
    J->hist = !J->pass && temporal && n->histOn && !(has(offsetof(LmxxfNrFrameInfo, reset) + 4) && fi->reset) && !C->histReset && C->lastLaunched;
    if (!J->pass) applyCond(n, ctl);
    job->handle = J; job->private_output = n->out;
    if (job->struct_size >= sizeof(LmxxfNrJob)) job->first_pass_output = n->out;   // one network pass per job here: the first pass's answer is the answer
    return LMXXF_NR_OK;
}
static int32_t RecordInputs(void* c, void* j, void* cmdList) {
    clearErr(); CTX(c); JOB(j);
    if (J->state != LMXXF_NR_JOB_PREPARED) return fail(LMXXF_NR_INVALID_ARGUMENT, "RecordInputs: job not prepared");
    ID3D12GraphicsCommandList* cl = (ID3D12GraphicsCommandList*)cmdList; Net* n = J->net;
    if (!cl || !J->color) return fail(LMXXF_NR_INVALID_ARGUMENT, "RecordInputs: command list / colour missing");
    ID3D12GraphicsCommandList2* c2 = nullptr; if (FAILED(cl->QueryInterface(IID_PPV_ARGS(&c2)))) return fail(LMXXF_NR_FAILED, "RecordInputs: the list has no ID3D12GraphicsCommandList2 (WriteBufferImmediate)");
    c2->Release();
    UINT cb[CB_N]; constants(n, J, cb); bind(C, n, cl, descriptors(C, n, J), cb);
    barrier(cl, J->color, J->colorState, kSrv); if (J->motion) barrier(cl, J->motion, J->motionState, kSrv);
    run(C, cl, P_CPY, (n->CW + 7) / 8, (n->CH + 7) / 8);
    bool launch = !J->pass;
    if (launch && n->split) {   // nr_split: a HIP job spans n->split host jobs; each host job releases one part
        if (n->hipPart == 0) J->part = n->hipPart = 1; else if (n->hipPart < n->split) J->part = ++n->hipPart; else launch = false;
    }
    if (launch) {
        if (!J->seq) { J->seq = ++C->seq; cb[CB_SEQ] = J->seq; cl->SetComputeRoot32BitConstants(3, CB_N, cb, 0); }
        if (J->part <= 1) { if (g_set.enc == 2) run(C, cl, P_LAT, 1, 1); run(C, cl, P_IMP, (n->W + 7) / 8, (n->H + 7) / 8); if (n->histOn) run(C, cl, P_HWP, (n->padW + 7) / 8, (n->padH + 7) / 8); }
        barrier(cl, n->sig, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);   // input is in io: release the runtime's flag_in wait
        D3D12_WRITEBUFFERIMMEDIATE_PARAMETER wp = {n->sig->GetGPUVirtualAddress(), J->seq}; D3D12_WRITEBUFFERIMMEDIATE_MODE wm = D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT;
        c2->WriteBufferImmediate(1, &wp, &wm); barrier(cl, n->sig, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    }
    barrier(cl, J->color, kSrv, J->colorState); if (J->motion) barrier(cl, J->motion, kSrv, J->motionState);
    J->launched = false; J->state = LMXXF_NR_JOB_PRODUCER_SUBMITTED; J->color = J->motion = nullptr;   // the host's resources are not kept
    return LMXXF_NR_OK;
}
static int32_t enqueue(Ctx* C, Job* J, bool async) {
    if (J->state != LMXXF_NR_JOB_PRODUCER_SUBMITTED) return fail(LMXXF_NR_INVALID_ARGUMENT, "EnqueueHip: inputs not recorded");
    Net* n = J->net; J->async = async; J->state = LMXXF_NR_JOB_NR_ENQUEUED;
    if (!J->pass && !J->seq) return LMXXF_NR_OK;   // nr_split: every part of the HIP job is out already, this job shows the last edit
    if (J->pass) { C->passJobs++; C->lastLaunched = false; return LMXXF_NR_OK; }
    const int hv = J->hist ? g_set.hist : 0;
    const int r = n->split ? nr.enq3(n->nc[0], J->seq, n->split + 1, J->part == 1 ? hv : 0, J->part) : nr.enq2(n->nc[0], J->seq, 3, hv);
    if (r < 0) return fail(LMXXF_NR_FAILED, "EnqueueHip: nr_ext_enqueue failed (%d%s)", r, r == -2 ? ": output self-check failed on this GPU" : "");
    if (r == 1) { J->pass = true; C->passJobs++; C->lastLaunched = false; if (n->split) n->hipPart = 0; logf("EnqueueHip: runtime busy, job %u passes through", J->seq); return LMXXF_NR_OK; }
    J->launched = true; C->lastLaunched = true; C->histReset = false; C->nrJobs++;
    if (n->split && J->part == n->split) n->hipLast = J->seq;
    if (C->nrJobs <= 3 || C->nrJobs % 1000 == 0) logf("job %u launched (%llu network jobs, %llu passthrough)", J->seq, C->nrJobs, C->passJobs);
    return LMXXF_NR_OK;
}
static int32_t EnqueueHipAsync(void* c, void* j, void*) { clearErr(); CTX(c); JOB(j); return enqueue(C, J, true); }
static int32_t EnqueueHip(void* c, void* j, void*) { clearErr(); CTX(c); JOB(j); return enqueue(C, J, false); }   // no D3D12 queue wait exists here; RecordOutputs' spin covers it
static int32_t ExecuteAfterProducer(void* c, void* j, void* q) { return EnqueueHip(c, j, q); }
static int32_t RecordOutputs(void* c, void* j, void* cmdList) {
    clearErr(); CTX(c); JOB(j);
    if (J->state != LMXXF_NR_JOB_NR_ENQUEUED && J->state != LMXXF_NR_JOB_NR_COMPLETE) return fail(LMXXF_NR_INVALID_ARGUMENT, "RecordOutputs: job not launched");
    ID3D12GraphicsCommandList* cl = (ID3D12GraphicsCommandList*)cmdList; Net* n = J->net; if (!cl) return fail(LMXXF_NR_INVALID_ARGUMENT, "RecordOutputs: no command list");
    if (n != C->net) return fail(LMXXF_NR_FAILED, "RecordOutputs: the network was rebuilt since this job (abandon it instead)");
    UINT cb[CB_N]; constants(n, J, cb); unsigned dn = 0, pr = 0; if (n->state == 1) nr.status(n->nc[0], &dn, &pr, nullptr);
    int ok = 2, spin = 0;
    if (n->split && !J->pass) {   // never waits: the newest finished HIP job's edit
        bool cap = n->hipPart == n->split && n->hipLast && seqReached(pr, n->hipLast);
        if (cap && !seqReached(dn, n->hipLast)) { cap = false; n->hipPart = 0; }   // dropped (flag_in timeout): start over
        if (cap) n->hipPart = 0;
        ok = cap || n->editValid ? 3 : 2; J->hist = false;
        cb[CB_OK] = ok; bind(C, n, cl, descriptors(C, n, nullptr), cb);
        if (cap) { run(C, cl, P_CAP, (n->W + 7) / 8, (n->H + 7) / 8); n->editValid = true; }
    } else {
        if (J->launched) { if (seqReached(dn, J->seq)) ok = 1; else if (seqReached(pr, J->seq)) ok = 2; else { ok = 0; spin = 1; } }
        cb[CB_OK] = ok; bind(C, n, cl, descriptors(C, n, nullptr), cb);
        if (spin) run(C, cl, P_SPIN, 1, 1);
    }
    barrier(cl, n->out, kSrv, kUav); run(C, cl, P_EXP, (n->CW + 7) / 8, (n->CH + 7) / 8); barrier(cl, n->out, kUav, kSrv);
    J->state = LMXXF_NR_JOB_CONSUMER_COMPLETE; return LMXXF_NR_OK;
}
static int32_t OutputReady(void* c, void* j, uint32_t* ready) {
    clearErr(); CTX(c); JOB(j); if (!ready) return fail(LMXXF_NR_INVALID_ARGUMENT, "OutputReady: null");
    *ready = 1; if (!J->launched || !J->async || J->net->split || J->state >= LMXXF_NR_JOB_CONSUMER_COMPLETE) return LMXXF_NR_OK;
    unsigned pr = 0; nr.status(J->net->nc[0], nullptr, &pr, nullptr); *ready = seqReached(pr, J->seq); return LMXXF_NR_OK;
}
static int32_t Poll(void* c, void* j, uint32_t* state) {
    clearErr(); CTX(c); JOB(j); if (!state) return fail(LMXXF_NR_INVALID_ARGUMENT, "Poll: null");
    if (J->state == LMXXF_NR_JOB_NR_ENQUEUED) { unsigned pr = 0; if (!J->launched || (!nr.status(J->net->nc[0], nullptr, &pr, nullptr) && seqReached(pr, J->seq))) J->state = LMXXF_NR_JOB_NR_COMPLETE; }
    *state = J->state; return LMXXF_NR_OK;
}
static int32_t Retire(void* c, void* j) { clearErr(); CTX(c); JOB(j); J->state = LMXXF_NR_JOB_RETIRED; return LMXXF_NR_OK; }
static int32_t AbandonJob(void* c, void* j) {
    clearErr(); CTX(c); JOB(j); const int r = waitDone(C, J, 2000); J->state = LMXXF_NR_JOB_RETIRED; C->histReset = true;
    return r ? fail(LMXXF_NR_FAILED, "AbandonJob: the job did not finish within 2 s") : LMXXF_NR_OK;
}
static int32_t CancelUnsubmitted(void* c, void* j) {   // inputs recorded but the list never ran / the job never launched
    clearErr(); CTX(c); JOB(j); if (J->launched) waitDone(C, J, 2000); J->state = LMXXF_NR_JOB_RETIRED; C->histReset = true; C->lastLaunched = false;
    if (J->net && J->net->split && J->part) J->net->hipPart = 0;   // a part that never runs: that HIP job is lost
    return LMXXF_NR_OK;
}
static int32_t ResetHistory(void* c) { clearErr(); CTX(c); C->histReset = true; return LMXXF_NR_OK; }
static int32_t Drain(void* c) { clearErr(); CTX(c); for (auto& j : C->jobs) if (waitDone(C, &j, 2000)) return fail(LMXXF_NR_FAILED, "Drain: a job did not finish within 2 s"); return LMXXF_NR_OK; }
static int32_t GetStatus(void* c, char* buf, uint32_t n) {
    Ctx* C = (Ctx*)c; if (!C || !buf || !n) return LMXXF_NR_INVALID_ARGUMENT; std::lock_guard<std::mutex> g(C->mu); Net* t = C->net;
    snprintf(buf, n, "rdna2-hip model=%s net=%s%s passes=%d hist=%s split=%d jobs=%llu pass=%llu skip=none%s", C->modelId.c_str(), t ? (t->state == 1 ? "" : t->state ? "failed:" : "loading:") : "none",
             t ? (std::to_string(t->W) + "x" + std::to_string(t->H)).c_str() : "", t ? t->passes : 0, t && t->histOn ? "on" : "off", t ? t->split : 0, C->nrJobs, C->passJobs, C->vkBridge ? " vk" : "");
    clearErr(); return LMXXF_NR_OK;
}
static int32_t GetLastErrorText(char* buf, uint32_t n) { if (!buf || !n) return LMXXF_NR_INVALID_ARGUMENT; std::lock_guard<std::mutex> g(g_errMu); snprintf(buf, n, "%s", g_err.c_str()); return LMXXF_NR_OK; }

extern "C" LMXXF_NR_EXPORT int32_t LmxxfNrGetApi(uint32_t abi, LmxxfNrApi* o) {
    std::call_once(g_setOnce, loadSettings);
    if (abi != LMXXF_NR_ABI_VERSION) return fail(LMXXF_NR_UNSUPPORTED_ABI, "LmxxfNrGetApi: ABI %u (this runtime speaks 1)", abi);
    if (!o || o->struct_size < sizeof(LmxxfNrApi)) return fail(LMXXF_NR_INVALID_ARGUMENT, "LmxxfNrGetApi: struct_size");
    const uint32_t ss = o->struct_size; memset(o, 0, sizeof(LmxxfNrApi)); o->struct_size = ss; o->abi_version = LMXXF_NR_ABI_VERSION;
    o->QueryCapabilities = QueryCapabilities; o->Create = Create; o->Destroy = Destroy; o->PrepareSession = PrepareSession; o->PrepareFrame = PrepareFrame;
    o->RecordInputs = RecordInputs; o->EnqueueHip = EnqueueHip; o->RecordOutputs = RecordOutputs; o->ExecuteAfterProducer = ExecuteAfterProducer;
    o->CancelUnsubmitted = CancelUnsubmitted; o->Poll = Poll; o->Retire = Retire; o->ResetHistory = ResetHistory; o->Drain = Drain; o->GetStatus = GetStatus;
    o->GetLastError = GetLastErrorText;
    return LMXXF_NR_OK;
}
// Optional exports (found with GetProcAddress by AMDNR)
// Test hook: the private copy of the colour the last jobs used and the network output are internal; tests read private_output only.

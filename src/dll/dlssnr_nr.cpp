// Drop-in nvngx.dll_dlssnr.dll that runs the RDNA2 DLSS-NR port through nr_runtime.dll (HIP).
// Same 12 exports as the shipped vendor-neutral port (see ../INTERFACE.md). Fail-safe: anything unexpected -> evaluate
// returns 0 ("not run") and the host keeps the native frame.
//
// Sync design (default, sync=1; DESIGN.md): evaluate N records into the host's list: colour -> shared io buffer (compute),
// WriteBufferImmediate(flag_in = N), a one-thread bounded spin on flag_out >= N, then io output -> `output` (compute).
// The HIP stream (queued at evaluate time) waits on flag_in, runs the network on io, writes flag_out. No CPU readback, no lag.
// If interop is unavailable (old runtime, import fails, no ID3D12GraphicsCommandList2) the async path below is used.
//
// Async design (fallback; forced by the host: evaluate only RECORDS into `cmd`, the host executes it later, so we cannot read the
// frame back synchronously and the network takes ~143 ms anyway):
//   evaluate N : if the worker is idle, record color -> readback copy (job). If a finished result exists, record
//                result -> output copy and return 1 (the result is stale by the network latency, ~10 frames at 60 fps).
//   worker     : create the HIP runtime once (seconds), then per job: wait until the readback copy landed (sentinel
//                polling, no queue access needed), run nr_run, publish the answer into an upload buffer.
// ponytail: single feature/pass; model size = one step dir (nr_data\\WxH) picked at create, fixed until the host recreates.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include "dlssnr_iface.h"
#include "r11g10.h"
#include <cmath>
#include <algorithm>
#include <d3dcommon.h>

// ---------------------------------------------------------------- logging / settings
static CRITICAL_SECTION g_cs; static FILE* g_log; static char g_err[512] = "";
static std::wstring g_dir;                         // folder of this DLL
static std::string g_dataDir, g_modelRes = "auto", g_logName = "dlssnr_nr.log"; static int g_vitEvery = 1, g_settleMs = 20, g_jobTimeoutMs = 3000;
static std::atomic<int> g_evalCalls{0};
static float g_intensity = -1.0f;                       // nr_set_intensity 0..2 (runtime strength of the network); <0 = keep the strength baked into the data dir
static float curIntensity() { return g_intensity; }
static std::atomic<bool> g_exiting{false};              // set by atexit: at process exit never join/destroy anything (leak it, the OS reclaims)
static struct ExitHook { ExitHook() { atexit([] { g_exiting = true; }); } } g_exitHook;
static int g_apply = 1, g_gate = 1; static float g_gain = 1.0f, g_gateLo = 0.03f, g_gateHi = 0.15f;   // apply=1: out = current frame + gain * (NR(old frame) - old frame) (no lag); apply=0: copy the finished (stale) NR image
static int g_sync = 1, g_spinMs = 250, g_maxInflight = 3, g_fg = 1, g_fgWaitMs = 4, g_fgMaxAge = 3;   // sync=1: same-frame NR through D3D12<->HIP shared memory (see DESIGN.md); falls back to the async path when unavailable

static void logf(const char* fmt, ...) {
    EnterCriticalSection(&g_cs);
    if (g_log) { va_list a; va_start(a, fmt); vfprintf(g_log, fmt, a); va_end(a); fputc('\n', g_log); fflush(g_log); }
    LeaveCriticalSection(&g_cs);
}
static void seterr(const char* fmt, ...) { va_list a; va_start(a, fmt); vsnprintf(g_err, sizeof g_err, fmt, a); va_end(a); logf("ERROR: %s", g_err); }
static std::string narrow(const std::wstring& w) { int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, 0, 0, 0, 0); std::string s(n ? n - 1 : 0, 0); if (n) WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, &s[0], n, 0, 0); return s; }

static void loadSettings() {
    std::wstring ini = g_dir + L"nr_port.ini"; char b[1024];
    GetPrivateProfileStringA("nr", "data_dir", "nr_data", b, sizeof b, narrow(ini).c_str()); g_dataDir = b;
    if (g_dataDir.size() < 2 || (g_dataDir[1] != ':' && g_dataDir[0] != '\\' && g_dataDir[0] != '/')) g_dataDir = narrow(g_dir) + g_dataDir;   // relative -> next to the DLL
    GetPrivateProfileStringA("nr", "model_res", "auto", b, sizeof b, narrow(ini).c_str()); g_modelRes = b;   // WxH (a subfolder of data_dir) or auto
    GetPrivateProfileStringA("nr", "log", "dlssnr_nr.log", b, sizeof b, narrow(ini).c_str()); g_logName = b;
    g_vitEvery = (int)GetPrivateProfileIntA("nr", "vit_every", 1, narrow(ini).c_str());
    g_settleMs = (int)GetPrivateProfileIntA("nr", "settle_ms", 20, narrow(ini).c_str());
    g_jobTimeoutMs = (int)GetPrivateProfileIntA("nr", "job_timeout_ms", 3000, narrow(ini).c_str());
    g_apply = (int)GetPrivateProfileIntA("nr", "apply", 1, narrow(ini).c_str());
    GetPrivateProfileStringA("nr", "gain", "1.0", b, sizeof b, narrow(ini).c_str()); g_gain = (float)atof(b);
    g_gate = (int)GetPrivateProfileIntA("nr", "gate", 1, narrow(ini).c_str());
    GetPrivateProfileStringA("nr", "gate_lo", "0.03", b, sizeof b, narrow(ini).c_str()); g_gateLo = (float)atof(b);
    GetPrivateProfileStringA("nr", "gate_hi", "0.15", b, sizeof b, narrow(ini).c_str()); g_gateHi = (float)atof(b);
    GetPrivateProfileStringA("nr", "intensity", "-1", b, sizeof b, narrow(ini).c_str()); g_intensity = (float)atof(b); if (!(g_intensity >= 0.f)) g_intensity = -1.f; if (g_intensity > 2.f) g_intensity = 2.f;
    g_sync = (int)GetPrivateProfileIntA("nr", "sync", 1, narrow(ini).c_str());
    g_spinMs = (int)GetPrivateProfileIntA("nr", "sync_timeout_ms", 250, narrow(ini).c_str()); g_spinMs = g_spinMs < 10 ? 10 : g_spinMs > 1000 ? 1000 : g_spinMs;   // GPU wait bound, far below the 2 s TDR
    g_fg = (int)GetPrivateProfileIntA("nr", "fg_mode", 1, narrow(ini).c_str());   // 1: frame-generation friendly sync (never native once an NR result exists; short bounded queue wait)
    g_fgWaitMs = (int)GetPrivateProfileIntA("nr", "fg_wait_ms", 4, narrow(ini).c_str()); g_fgWaitMs = g_fgWaitMs < 0 ? 0 : g_fgWaitMs > 1000 ? 1000 : g_fgWaitMs;
    g_fgMaxAge = (int)GetPrivateProfileIntA("nr", "fg_max_age", 3, narrow(ini).c_str()); g_fgMaxAge = g_fgMaxAge < 0 ? 0 : g_fgMaxAge > 60 ? 60 : g_fgMaxAge;   // fg: reuse an NR edit at most this many frames, fading out
    g_maxInflight = (int)GetPrivateProfileIntA("nr", "max_inflight", g_fg ? 1 : 3, narrow(ini).c_str()); g_maxInflight = g_maxInflight < 1 ? 1 : g_maxInflight > 8 ? 8 : g_maxInflight;
    auto cl = [](float v, float lo, float hi) { return v != v ? lo : v < lo ? lo : v > hi ? hi : v; };   // ini garbage (NaN, huge, negative) never reaches the GPU constants
    g_gain = cl(g_gain, 0.f, 2.f); g_gateLo = cl(g_gateLo, 0.f, 1.f); g_gateHi = cl(g_gateHi, g_gateLo, 1.f);
    g_settleMs = g_settleMs < 0 ? 0 : g_settleMs > 500 ? 500 : g_settleMs; g_jobTimeoutMs = g_jobTimeoutMs < 200 ? 200 : g_jobTimeoutMs > 60000 ? 60000 : g_jobTimeoutMs; if (g_vitEvery < 1) g_vitEvery = 1;
    std::string lp = (g_logName.size() > 1 && g_logName[1] == ':') ? g_logName : narrow(g_dir) + g_logName;
    fopen_s(&g_log, lp.c_str(), "a");
}
BOOL WINAPI DllMain(HINSTANCE h, DWORD why, LPVOID) {
    if (why == DLL_PROCESS_ATTACH) {
        InitializeCriticalSection(&g_cs);
        wchar_t p[MAX_PATH]; GetModuleFileNameW(h, p, MAX_PATH); g_dir = p; g_dir = g_dir.substr(0, g_dir.find_last_of(L'\\') + 1);
    }
    return TRUE;
}

// ---------------------------------------------------------------- nr_runtime (loaded dynamically)
struct nr_ctx;
typedef nr_ctx* (*P_create)(const char*, int, int, int, char*, int);
typedef int (*P_run)(nr_ctx*, const void*, void*, float*);
typedef void (*P_destroy)(nr_ctx*);
typedef int (*P_intensity)(nr_ctx*, float);
static P_create p_create; static P_run p_run; static P_destroy p_destroy; static P_intensity p_intensity;
typedef int (*P_extOpen)(nr_ctx*, void*, unsigned long long, unsigned long long, unsigned long long, unsigned long long, void*, unsigned long long, unsigned long long, char*, int);
typedef int (*P_extEnqueue)(nr_ctx*, unsigned, int);
static P_extOpen p_extOpen; static P_extEnqueue p_extEnqueue;   // absent in an old nr_runtime.dll -> async path
static bool loadRuntime() {
    if (p_create) return true;
    std::wstring p = g_dir + L"nr_runtime.dll";
    HMODULE m = LoadLibraryExW(p.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);   // its HIP dependencies are searched beside it too
    if (!m) { seterr("cannot load nr_runtime.dll (error %lu) - is the HIP runtime (amdhip64_*.dll) on PATH or beside it?", GetLastError()); return false; }
    p_create = (P_create)GetProcAddress(m, "nr_create"); p_run = (P_run)GetProcAddress(m, "nr_run"); p_destroy = (P_destroy)GetProcAddress(m, "nr_destroy");
    if (!p_create || !p_run || !p_destroy) { p_create = nullptr; seterr("nr_runtime.dll lacks nr_create/nr_run/nr_destroy"); return false; }
    p_extOpen = (P_extOpen)GetProcAddress(m, "nr_ext_open"); p_extEnqueue = (P_extEnqueue)GetProcAddress(m, "nr_ext_enqueue");
    p_intensity = (P_intensity)GetProcAddress(m, "nr_set_intensity");            // absent in an old nr_runtime.dll: the baked strength stays
    if (!p_intensity && curIntensity() >= 0.f) logf("nr_runtime.dll has no nr_set_intensity (old runtime): intensity is ignored");
    return true;
}
// SEH wrappers: an access violation inside the HIP runtime must fail the feature, never the game (worker thread has no other guard)
static nr_ctx* safeCreate(const char* d, int w, int h, int v, char* e, int n) { __try { return p_create(d, w, h, v, e, n); } __except (EXCEPTION_EXECUTE_HANDLER) { snprintf(e, n, "access violation 0x%08lx in nr_create", (unsigned long)GetExceptionCode()); return nullptr; } }
static int safeRun(nr_ctx* c, const void* i, void* o, float* ms) { __try { return p_run(c, i, o, ms); } __except (EXCEPTION_EXECUTE_HANDLER) { return -1000; } }
static void sanitizeHalf4(uint16_t* p, size_t n) {   // NaN -> 0, +Inf -> 65504, -Inf -> 0 (k_import turns +Inf into NaN; unlanded readback rows are 0xFF.. = NaN)
    for (size_t i = 0; i < n; i++) { uint16_t h = p[i]; if ((h & 0x7C00) == 0x7C00) p[i] = ((h & 0x3FF) || (h & 0x8000)) ? 0 : 0x7BFF; }
}

// ---------------------------------------------------------------- half helpers
static float h2f(uint16_t h) { int e = (h >> 10) & 31, m = h & 1023; float v = e == 0 ? std::ldexp((float)m, -24) : e == 31 ? INFINITY : std::ldexp((float)(1024 + m), e - 25); return (h & 0x8000) ? -v : v; }
static uint16_t f2h(float f) {                              // round to nearest half, clamp to +-65504
    if (!(f == f)) return 0; if (f > 65504.f) f = 65504.f; if (f < -65504.f) f = -65504.f;
    uint32_t b; memcpy(&b, &f, 4); uint16_t sgn = (uint16_t)((b >> 16) & 0x8000); b &= 0x7FFFFFFF; float a; memcpy(&a, &b, 4);
    if (a < 6.1035156e-5f) return sgn | (uint16_t)lrintf(a * 16777216.f);                  // subnormal half: units of 2^-24
    int e; float m = frexpf(a, &e); uint32_t mant = (uint32_t)lrintf((m * 2.f - 1.f) * 1024.f); int he = e - 1 + 15;   // a = m*2^e, m in [0.5,1)
    if (mant == 1024) { mant = 0; he++; } if (he >= 31) return sgn | 0x7BFF; return sgn | (uint16_t)((he << 10) | mant);
}

// ---------------------------------------------------------------- delta compute pass
static const char* kHlsl =
"Texture2D<float4> col : register(t0);\n"
"ByteAddressBuffer edit : register(t1);\n"
"ByteAddressBuffer oldc : register(t2);\n"
"RWTexture2D<float4> outp : register(u0);\n"
"cbuffer C : register(b0) { uint W; uint H; uint pitch; uint gateOn; float g0; float g1; };\n"
"[numthreads(8,8,1)] void main(uint3 id : SV_DispatchThreadID) {\n"
"  if (id.x >= W || id.y >= H) return;\n"
"  float4 c = col[id.xy]; uint2 r = edit.Load2(id.y * pitch + id.x * 8);\n"
"  float3 e = float3(f16tof32(r.x & 0xFFFF), f16tof32(r.x >> 16), f16tof32(r.y & 0xFFFF));\n"
"  float w = 1.0;\n"
"  if (gateOn != 0) { uint2 q = oldc.Load2(id.y * pitch + id.x * 8); float3 o = float3(f16tof32(q.x & 0xFFFF), f16tof32(q.x >> 16), f16tof32(q.y & 0xFFFF));\n"
"    float d = max(abs(c.r - o.r), max(abs(c.g - o.g), abs(c.b - o.b))); w = (d < 1e30) ? 1.0 - smoothstep(g0, g1, d) : 1.0; }\n"
"  outp[id.xy] = float4(max(c.rgb + e * w, 0.0), c.a);\n"
"}\n";
static const int kRing = 8, kDesc = 4;                                 // descriptor sets in flight

// ---------------------------------------------------------------- sync path shaders (one source, three entry points)
// imp : game colour (any of the 3 formats, the SRV converts) -> sanitized half4 network input in the shared io buffer
// spin: one thread polls the flag the HIP stream writes when frame `seq` is done; bounded (TDR-safe), result -> st[0]
// exp : NR result (or the live frame if spin timed out) -> output texture (the UAV converts to the game's format)
static const char* kSyncHlsl =
"Texture2D<float4> col : register(t0);\n"
"RWTexture2D<float4> outp : register(u0);\n"
"globallycoherent RWByteAddressBuffer io : register(u1);\n"
"globallycoherent RWByteAddressBuffer st : register(u2);\n"
"cbuffer C : register(b0) { uint W; uint H; uint seq; uint maxIt; uint outOff; uint flagOff; float gain; uint FWH; uint minSeq; float gLo; float gHi; uint gate; uint frame; uint maxAge; };\n"   // minSeq != 0: fg mode, reuse the edit captured from a frame >= minSeq;   // W,H model; FWH = frame W | H << 16
"RWByteAddressBuffer ob : register(u3);\n"   // PACKED: frame-size R9G9B9E5 staging, copied into the output texture afterwards
"uint e5(float3 c) {\n"   // float3 -> R9G9B9E5_SHAREDEXP (N=9, B=15)
"  c = clamp(c, 0.0, 65408.0); float m = max(c.r, max(c.g, c.b)); int e = max(-16, (int)floor(log2(max(m, 1e-30)))) + 16;\n"
"  float d = exp2(e - 24); if (floor(m / d + 0.5) >= 512.0) { d *= 2.0; e++; } uint3 q = (uint3)floor(c / d + 0.5);\n"
"  return q.r | (q.g << 9) | (q.b << 18) | ((uint)e << 27); }\n"
"void put(uint2 p, float4 v) {\n"
"#ifdef PACKED\n"
"  ob.Store(p.y * ((((FWH & 0xFFFF) * 4) + 255) & ~255) + p.x * 4, e5(v.rgb));\n"
"#else\n"
"  outp[p] = v;\n"
"#endif\n"
"}\n"   // W,H model; FWH = frame W | H << 16
"float san(float v) { return (v != v || v < -65504.0) ? 0.0 : min(v, 65504.0); }\n"   // NaN/-Inf -> 0, +Inf -> 65504 (as the CPU path)
"uint pk(float a, float b) { return f32tof16(san(a)) | (f32tof16(san(b)) << 16); }\n"
"float4 bil(float2 p, uint off) {\n"   // bilinear fetch of a half4 model-size image in io at model coords p (pixel centres at .5)
"  p = clamp(p - 0.5, 0.0, float2(W - 1, H - 1)); uint2 a = (uint2)p; uint2 b = min(a + 1, uint2(W - 1, H - 1)); float2 t = p - a;\n"
"  uint2 q[4] = { io.Load2(off + (a.y * W + a.x) * 8), io.Load2(off + (a.y * W + b.x) * 8), io.Load2(off + (b.y * W + a.x) * 8), io.Load2(off + (b.y * W + b.x) * 8) };\n"
"  float4 v[4]; for (int i = 0; i < 4; i++) v[i] = float4(f16tof32(q[i].x & 0xFFFF), f16tof32(q[i].x >> 16), f16tof32(q[i].y & 0xFFFF), f16tof32(q[i].y >> 16));\n"
"  return lerp(lerp(v[0], v[1], t.x), lerp(v[2], v[3], t.x), t.y); }\n"
"float4 sn(float4 v) { return float4(san(v.x), san(v.y), san(v.z), san(v.w)); }\n"
"float4 colAt(float2 p) {\n"           // frame colour averaged over the model pixel footprint (4 bilinear taps), p = frame coords of the centre
"  uint FW = FWH & 0xFFFF, FH = FWH >> 16; float2 s = float2(FW, FH) / float2(W, H) * 0.25; float4 r = 0;\n"
"  for (int j = 0; j < 4; j++) { float2 q = clamp(p + s * float2((j & 1) ? 1 : -1, (j & 2) ? 1 : -1) - 0.5, 0.0, float2(FW - 1, FH - 1));\n"
"    uint2 a = (uint2)q; uint2 b = min(a + 1, uint2(FW - 1, FH - 1)); float2 t = q - a;\n"
"    r += lerp(lerp(sn(col[a]), sn(col[uint2(b.x, a.y)]), t.x), lerp(sn(col[uint2(a.x, b.y)]), sn(col[b]), t.x), t.y); }\n"
"  return r * 0.25; }\n"
"[numthreads(8,8,1)] void imp(uint3 id : SV_DispatchThreadID) {\n"
"  if (id.x >= W || id.y >= H) return; uint FW = FWH & 0xFFFF, FH = FWH >> 16;\n"
"  float4 c = (FW == W && FH == H) ? col[id.xy] : colAt((id.xy + 0.5) * float2(FW, FH) / float2(W, H));\n"
"  io.Store2((id.y * W + id.x) * 8, uint2(pk(c.r, c.g), pk(c.b, c.a))); }\n"
"[numthreads(1,1,1)] void spin() {\n"
"  uint v = 0; for (uint i = 0; i < maxIt; i++) { io.InterlockedOr(flagOff, 0, v); if ((int)(v - seq) >= 0) break; }\n"
"  st.Store(0, (int)(v - seq) >= 0 ? 1u : 0u); }\n"
"[numthreads(8,8,1)] void cap(uint3 id : SV_DispatchThreadID) {\n"   // fg: keep the NR edit (out - in, model size) of a finished frame at st+256
"  if (id.x >= W || id.y >= H || st.Load(0) == 0) return; uint o = (id.y * W + id.x) * 8;\n"
"  float4 i = bil(id.xy + 0.5, 0), d = sn(bil(id.xy + 0.5, outOff) - i); st.Store2(256 + o, uint2(pk(d.r, d.g), pk(d.b, 0)));\n"
"  st.Store2(256 + W * H * 8 + o, uint2(pk(i.r, i.g), pk(i.b, 0)));\n"   // + the model input the edit belongs to (motion gate)
"  if (id.x == 0 && id.y == 0) { st.Store(4, seq); st.Store(8, frame); } }\n"
"[numthreads(8,8,1)] void exp(uint3 id : SV_DispatchThreadID) {\n"
"  uint FW = FWH & 0xFFFF, FH = FWH >> 16; if (id.x >= FW || id.y >= FH) return; float4 c = col[id.xy];\n"
"  if (st.Load(0) == 0) {\n"
"    uint age = frame - st.Load(8);\n"   // fg: frames since the edit's own frame; older than maxAge -> live frame only
"    if (minSeq != 0 && (int)(st.Load(4) - minSeq) >= 0 && age <= maxAge) { float2 p = (id.xy + 0.5) * float2(W, H) / float2(FW, FH);\n"   // fg: live frame + last NR edit
"      p = clamp(p - 0.5, 0.0, float2(W - 1, H - 1)); uint2 a = (uint2)p; uint2 b = min(a + 1, uint2(W - 1, H - 1)); float2 t = p - a; float3 e[4]; uint2 ix[4] = { a, uint2(b.x, a.y), uint2(a.x, b.y), b };\n"
"      for (int i = 0; i < 4; i++) { uint2 q = st.Load2(256 + (ix[i].y * W + ix[i].x) * 8); e[i] = float3(f16tof32(q.x & 0xFFFF), f16tof32(q.x >> 16), f16tof32(q.y & 0xFFFF));\n"
"        if (gate) { uint2 u = st.Load2(256 + W * H * 8 + (ix[i].y * W + ix[i].x) * 8); float3 o = float3(f16tof32(u.x & 0xFFFF), f16tof32(u.x >> 16), f16tof32(u.y & 0xFFFF));\n"   // motion gate per model pixel: live colour (sampled as imp does) vs the edit's input
"          float3 l = ((FW == W && FH == H) ? sn(col[ix[i]]) : colAt((ix[i] + 0.5) * float2(FW, FH) / float2(W, H))).rgb; float3 a = abs(l - o);\n"
"          e[i] *= 1.0 - smoothstep(gLo, gHi, max(a.r, max(a.g, a.b))); } }\n"
"      float3 d = lerp(lerp(e[0], e[1], t.x), lerp(e[2], e[3], t.x), t.y) * (1.0 - (float)age / (maxAge + 1)); put(id.xy, float4(max(c.rgb + gain * d, 0.0), c.a)); return; }\n"
"    put(id.xy, c); return; }\n"
"  if (FW != W || FH != H) {\n"     // resampled: carry the model's change (out - in, model size) up to the frame
"    float2 p = (id.xy + 0.5) * float2(W, H) / float2(FW, FH); float4 d = sn(bil(p, outOff) - bil(p, 0));\n"
"    put(id.xy, float4(max(c.rgb + gain * d.rgb, 0.0), c.a)); return; }\n"                       // timeout: native frame
"  uint2 r = io.Load2(outOff + (id.y * W + id.x) * 8);\n"
"  float3 e = float3(san(f16tof32(r.x & 0xFFFF)), san(f16tof32(r.x >> 16)), san(f16tof32(r.y & 0xFFFF)));\n"
"  float3 o = gain == 1.0 ? e : c.rgb + gain * (e - c.rgb);\n"
"  put(id.xy, float4(max(o, 0.0), c.a)); }\n";
static const int kSyncRing = 16;                                       // descriptor pairs (colour SRV, output UAV) in flight

// ---------------------------------------------------------------- feature
enum { S_LOADING = 0, S_IDLE = 1, S_WAIT_COPY = 2, S_RUNNING = 3, S_FAILED = 4, S_SYNC = 5 };
struct Sync {                                       // D3D12 side of the sync path; HIP imports io + sig
    ID3D12Resource *io = nullptr, *sig = nullptr, *st = nullptr; HANDLE hIo = nullptr, hSig = nullptr;
    UINT64 ioAlloc = 0, ioSize = 0, sigAlloc = 0, sigSize = 0, outOff = 0, flagOff = 0;
    ID3D12RootSignature* rs = nullptr; ID3D12PipelineState *imp = nullptr, *spin = nullptr, *exp = nullptr, *cap = nullptr, *expPk = nullptr; ID3D12Resource* ob = nullptr; unsigned obW = 0, obH = 0; ID3D12DescriptorHeap* heap = nullptr; UINT incr = 0;
    unsigned seq = 0; unsigned busy = 0; unsigned ran = 0, rec = 0, reused = 0, minSeq = 1; unsigned frm = 0, frmOf[16] = {};   // fg: frame counter, frame of each in-flight seq (edit age)
};
struct Feature {
    unsigned W = 0, H = 0, pitch = 0;
    ID3D12Resource* rb = nullptr; ID3D12Resource* up[2] = {}; uint8_t* rbp = nullptr; uint8_t* upp[2] = {};
    std::atomic<int> state{S_LOADING}; std::atomic<int> published{-1}; std::atomic<bool> quit{false};
    std::thread th; std::string dir; nr_ctx* ctx = nullptr; DWORD jobStart = 0; bool warnedFmt = false, warnedSize = false;
    std::atomic<unsigned> resetGen{0}, slotEval[2] = {{0}, {0}}; unsigned jobGen = 0; float appliedIntensity = -2.f;   // resetGen: bumped by a real reset; slotEval: last evaluate that recorded a read of a slot
    std::vector<uint8_t> tin, tout;
    ID3D12Device* dev = nullptr; ID3D12RootSignature* rs = nullptr; ID3D12PipelineState* pso = nullptr; ID3D12DescriptorHeap* heap = nullptr; UINT incr = 0; bool delta = false;
    ID3D12Resource* upOld[2] = {}; uint8_t* upOldp[2] = {}; ID3D12Resource* zero = nullptr; uint8_t* zerop = nullptr;   // old model input per slot (motion gate); all-zero buffer for "no result yet"
    std::atomic<int> inFmt{(int)DXGI_FORMAT_R16G16B16A16_FLOAT}, outFmt{(int)DXGI_FORMAT_R16G16B16A16_FLOAT};   // game texture formats (set by evaluate)
    Sync* sy = nullptr;                                                 // non-null: sync resources exist (worker tries nr_ext_open)
};
static const DXGI_FORMAT kFmt = DXGI_FORMAT_R16G16B16A16_FLOAT;
static bool fmtOk(DXGI_FORMAT f) { return f == DXGI_FORMAT_R16G16B16A16_FLOAT || f == DXGI_FORMAT_R11G11B10_FLOAT || f == DXGI_FORMAT_R8G8B8A8_UNORM; }
static unsigned fmtBpp(int f) { return f == DXGI_FORMAT_R16G16B16A16_FLOAT ? 8 : 4; }
static unsigned align256(unsigned v) { return (v + 255) & ~255u; }

static ID3D12Resource* makeBuffer(ID3D12Device* d, D3D12_HEAP_TYPE type, UINT64 size, uint8_t** map) {
    D3D12_HEAP_PROPERTIES hp = {}; hp.Type = type;
    D3D12_RESOURCE_DESC rd = {}; rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; rd.Width = size; rd.Height = 1; rd.DepthOrArraySize = 1;
    rd.MipLevels = 1; rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* r = nullptr;
    if (FAILED(d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, type == D3D12_HEAP_TYPE_READBACK ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&r)))) return nullptr;
    void* p = nullptr; D3D12_RANGE none = {0, 0};
    if (FAILED(r->Map(0, type == D3D12_HEAP_TYPE_READBACK ? nullptr : &none, &p))) { r->Release(); return nullptr; }
    *map = (uint8_t*)p; return r;
}

static bool sampledDone(Feature* F) {             // every sampled texel no longer the 0xFF.. sentinel => the copy landed
    const unsigned bpp = fmtBpp(F->inFmt);
    auto sentinel = [&](size_t off) { uint64_t v = 0; memcpy(&v, F->rbp + off, bpp); return bpp == 8 ? v == ~0ull : (uint32_t)v == ~0u; };
    if (F->inFmt == (int)DXGI_FORMAT_R8G8B8A8_UNORM) {   // 0xFFFFFFFF is a valid white pixel in RGBA8: any changed sample means the copy ran (settle_ms covers the rest)
        for (unsigned y = 0; y < F->H; y += 16) for (int side = 0; side < 2; side++) if (!sentinel((size_t)y * F->pitch + (side ? (F->W - 1) * bpp : 0))) return true;
        return !sentinel((size_t)(F->H - 1) * F->pitch + (F->W - 1) * bpp); }
    for (unsigned y = 0; y < F->H; y += 16)
        for (int side = 0; side < 2; side++) if (sentinel((size_t)y * F->pitch + (side ? (F->W - 1) * bpp : 0))) return false;
    return !sentinel((size_t)(F->H - 1) * F->pitch + (F->W - 1) * bpp);
}

// Input/output statistics of one job for the log: tells the first game session whether the value range the host hands the network is sane.
static void logStats(Feature* F) {
    std::vector<float> v; double d = 0; size_t nd = 0; int bad = 0;
    for (unsigned y = 0; y < F->H; y += 5) for (unsigned x = 0; x < F->W; x += 5) {
        const uint16_t* a = (const uint16_t*)(F->tin.data() + ((size_t)y * F->W + x) * 8); const uint16_t* b = (const uint16_t*)(F->tout.data() + ((size_t)y * F->W + x) * 8);
        if ((b[0] & 0x7C00) == 0x7C00 || (b[1] & 0x7C00) == 0x7C00 || (b[2] & 0x7C00) == 0x7C00) { bad++; continue; }
        v.push_back((h2f(a[0]) + h2f(a[1]) + h2f(a[2])) * (1.0f / 3.0f)); for (int c = 0; c < 3; c++) { d += fabs((double)h2f(b[c]) - (double)h2f(a[c])); nd++; } }
    if (v.empty()) { logf("stats: no finite samples (%d non-finite outputs)", bad); return; }
    std::sort(v.begin(), v.end()); double m = 0; for (float f : v) m += f; m /= v.size();
    logf("stats: model input mean %.4g p50 %.4g p99 %.4g max %.4g | network changes it by mean %.4f per channel (%zu samples, %d non-finite outputs)", m, v[v.size() / 2], v[v.size() * 99 / 100], v.back(), d / nd, v.size(), bad);
}

static void worker(Feature* F) {
    try {
        char err[512] = {0};
        if (!loadRuntime()) { F->state = S_FAILED; return; }
        logf("worker: nr_create(%s, %ux%u, vit_every %d)...", F->dir.c_str(), F->W, F->H, g_vitEvery);
        DWORD t0 = GetTickCount();
        F->ctx = safeCreate(F->dir.c_str(), (int)F->W, (int)F->H, g_vitEvery, err, sizeof err);
        if (!F->ctx) { seterr("nr_create failed: %s", err); F->state = S_FAILED; return; }
        if (F->quit) { if (p_destroy) p_destroy(F->ctx); F->ctx = nullptr; return; }   // released while the runtime was loading: clean up and leave
        logf("worker: runtime ready after %lu ms", GetTickCount() - t0);
        if (F->sy) {
            if (!p_extOpen || !p_extEnqueue) logf("sync: nr_runtime.dll has no nr_ext_* (old runtime) - async path");
            else if (p_extOpen(F->ctx, F->sy->hIo, F->sy->ioAlloc, F->sy->ioSize, F->sy->outOff, F->sy->flagOff, F->sy->hSig, F->sy->sigAlloc, F->sy->sigSize, err, sizeof err)) logf("sync: nr_ext_open failed (%s) - async path", err);
            else { if (p_intensity && curIntensity() >= 0.f && p_intensity(F->ctx, curIntensity()) == 0) logf("worker: intensity %.2f", curIntensity());
                   logf("sync: D3D12<->HIP interop ready, same-frame NR (GPU wait bound %d ms, max %d frames in flight)", g_spinMs, g_maxInflight); F->state = S_SYNC; return; }   // the worker has nothing more to do
        }
        F->state = S_IDLE;
        while (!F->quit) {
            if (F->state == S_WAIT_COPY) {
                if (GetTickCount() - F->jobStart > (DWORD)g_jobTimeoutMs) { logf("worker: copy never landed (frame not executed?), dropping job"); F->state = S_IDLE; continue; }
                if (!sampledDone(F)) { Sleep(1); continue; }
                Sleep(g_settleMs);                          // let the rest of the copy finish
                F->state = S_RUNNING;
                for (unsigned y = 0; y < F->H; y++) {
                    uint8_t* d = F->tin.data() + (size_t)y * F->W * 8; const uint8_t* sr = F->rbp + (size_t)y * F->pitch;
                    if (F->inFmt == (int)DXGI_FORMAT_R11G11B10_FLOAT) for (unsigned x = 0; x < F->W; x++) unpack_r11g11b10(((const uint32_t*)sr)[x], (uint16_t*)d + 4 * x);
                    else if (F->inFmt == (int)DXGI_FORMAT_R8G8B8A8_UNORM) for (unsigned x = 0; x < F->W * 4; x++) ((uint16_t*)d)[x] = f2h(sr[x] / 255.f);
                    else memcpy(d, sr, (size_t)F->W * 8); }
                sanitizeHalf4((uint16_t*)F->tin.data(), (size_t)F->W * F->H * 4);
                { const float want = curIntensity(); if (p_intensity && want >= 0.f && F->appliedIntensity != want) { if (p_intensity(F->ctx, want) == 0) logf("worker: intensity %.2f", want); F->appliedIntensity = want; } }
                float ms = 0; int r = safeRun(F->ctx, F->tin.data(), F->tout.data(), &ms);
                if (r) { seterr("nr_run failed (%d)", r); F->state = S_FAILED; return; }
                { static unsigned jobs = 0; if (jobs < 3 || jobs % 8 == 0) logStats(F); jobs++; }
                if (F->jobGen != F->resetGen) { logf("worker: result dropped (a reset/camera cut happened while it was computed)"); F->state = S_IDLE; continue; }
                int slot = F->published.load() < 0 ? 0 : 1 - F->published.load();
                { const DWORD tw = GetTickCount(); while (!F->quit && (unsigned)(g_evalCalls.load() - F->slotEval[slot].load()) < 6u && GetTickCount() - tw < 250) Sleep(2); }   // the GPU may still read this slot from a list recorded a few frames ago (deadline in ms: Sleep(1) is ~15 ms on Windows)
                for (unsigned y = 0; y < F->H; y++) {
                    uint8_t* d = F->upp[slot] + (size_t)y * F->pitch; const uint8_t* sr = F->tout.data() + (size_t)y * F->W * 8;
                    if (F->delta) {                                             // edit = NR(old) - old, half4 (rgb used)
                        const uint16_t* a = (const uint16_t*)sr; const uint16_t* b0 = (const uint16_t*)(F->tin.data() + (size_t)y * F->W * 8); uint16_t* e = (uint16_t*)d;
                        for (unsigned x = 0; x < F->W * 4; x++) e[x] = f2h(g_gain * (h2f(a[x]) - h2f(b0[x])));
                        memcpy(F->upOldp[slot] + (size_t)y * F->pitch, b0, (size_t)F->W * 8); }
                    else if (F->outFmt == (int)DXGI_FORMAT_R11G11B10_FLOAT) for (unsigned x = 0; x < F->W; x++) pack_r11g11b10((const uint16_t*)sr + 4 * x, (uint32_t*)d + x);
                    else if (F->outFmt == (int)DXGI_FORMAT_R8G8B8A8_UNORM) for (unsigned x = 0; x < F->W * 4; x++) { float v = h2f(((const uint16_t*)sr)[x]); d[x] = (uint8_t)(v <= 0.f ? 0 : v >= 1.f ? 255 : v * 255.f + .5f); }
                    else memcpy(d, sr, (size_t)F->W * 8); }
                F->published = slot; logf("worker: result published in slot %d (network %.1f ms GPU)", slot, ms);
                F->state = S_IDLE;
            } else Sleep(2);
        }
    } catch (...) { seterr("worker exception"); F->state = S_FAILED; }
}

static bool makeDelta(Feature* F) {
    typedef HRESULT (WINAPI *PFN_Compile)(const void*, SIZE_T, const char*, const void*, void*, const char*, const char*, UINT, UINT, ID3DBlob**, ID3DBlob**);
    HMODULE c = LoadLibraryExW(L"d3dcompiler_47.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32); /* System32 only: never a copy lying next to the game exe */ auto comp = c ? (PFN_Compile)GetProcAddress(c, "D3DCompile") : nullptr;
    if (!comp) { logf("delta: d3dcompiler_47.dll missing, falling back to copy"); return false; }
    ID3DBlob *cs = nullptr, *er = nullptr;
    if (FAILED(comp(kHlsl, strlen(kHlsl), "nr_delta", nullptr, nullptr, "main", "cs_5_0", 1 << 15, 0, &cs, &er))) { logf("delta: shader compile failed: %s", er ? (char*)er->GetBufferPointer() : "?"); return false; }
    D3D12_DESCRIPTOR_RANGE rg[4] = {}; for (int i = 0; i < 4; i++) { rg[i].RangeType = i == 3 ? D3D12_DESCRIPTOR_RANGE_TYPE_UAV : D3D12_DESCRIPTOR_RANGE_TYPE_SRV; rg[i].NumDescriptors = 1; rg[i].BaseShaderRegister = i == 3 ? 0 : i; rg[i].OffsetInDescriptorsFromTableStart = i; }
    D3D12_ROOT_PARAMETER rp[2] = {}; rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; rp[0].DescriptorTable.NumDescriptorRanges = 4; rp[0].DescriptorTable.pDescriptorRanges = rg;
    rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; rp[1].Constants.Num32BitValues = 6;
    D3D12_ROOT_SIGNATURE_DESC rsd = {}; rsd.NumParameters = 2; rsd.pParameters = rp;
    ID3DBlob *sb = nullptr, *se = nullptr;
    if (FAILED(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sb, &se)) || FAILED(F->dev->CreateRootSignature(0, sb->GetBufferPointer(), sb->GetBufferSize(), IID_PPV_ARGS(&F->rs)))) { logf("delta: root signature failed"); return false; }
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {}; pd.pRootSignature = F->rs; pd.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
    if (FAILED(F->dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&F->pso)))) { logf("delta: pipeline state failed"); return false; }
    D3D12_DESCRIPTOR_HEAP_DESC hd = {}; hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors = kDesc * kRing; hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(F->dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&F->heap)))) { logf("delta: descriptor heap failed"); return false; }
    F->incr = F->dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    return true;
}

static void freeSync(Sync* s) {
    if (!s) return;
    for (IUnknown* u : std::initializer_list<IUnknown*>{s->io, s->sig, s->st, s->rs, s->imp, s->spin, s->exp, s->cap, s->expPk, s->ob, s->heap}) if (u) u->Release();
    if (s->hIo) CloseHandle(s->hIo); if (s->hSig) CloseHandle(s->hSig); delete s;
}
static ID3D12Resource* sharedBuffer(ID3D12Device* d, UINT64 size, D3D12_HEAP_FLAGS hf, UINT64* alloc) {
    D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd = {}; rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; rd.Width = size; rd.Height = 1; rd.DepthOrArraySize = 1;
    rd.MipLevels = 1; rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ID3D12Resource* r = nullptr; if (FAILED(d->CreateCommittedResource(&hp, hf, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&r)))) return nullptr;
    if (alloc) *alloc = d->GetResourceAllocationInfo(0, 1, &rd).SizeInBytes; return r;
}
// D3D12 half of the sync path: shared io (input | output | flag) and sig (flag_in) buffers + NT handles, a private status word, 3 compute PSOs.
static Sync* makeSync(Feature* F) {
    typedef HRESULT (WINAPI *PFN_Compile)(const void*, SIZE_T, const char*, const void*, void*, const char*, const char*, UINT, UINT, ID3DBlob**, ID3DBlob**);
    HMODULE c = LoadLibraryExW(L"d3dcompiler_47.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32); auto comp = c ? (PFN_Compile)GetProcAddress(c, "D3DCompile") : nullptr;
    if (!comp) { logf("sync: d3dcompiler_47.dll missing"); return nullptr; }
    auto* s = new Sync; ID3D12Device* d = F->dev;
    const UINT64 n = (UINT64)F->W * F->H * 8, a = (n + 65535) & ~65535ull;
    s->outOff = a; s->flagOff = 2 * a; s->ioSize = 2 * a + 65536; s->sigSize = 65536;
    s->io = sharedBuffer(d, s->ioSize, D3D12_HEAP_FLAG_SHARED, &s->ioAlloc); s->sig = sharedBuffer(d, s->sigSize, D3D12_HEAP_FLAG_SHARED, &s->sigAlloc); s->st = sharedBuffer(d, 256 + 2 * n, D3D12_HEAP_FLAG_NONE, nullptr);   // status words | fg edit | its model input (half4, model size each)
    if (!s->io || !s->sig || !s->st) { logf("sync: cannot create shared buffers"); freeSync(s); return nullptr; }
    if (FAILED(d->CreateSharedHandle(s->io, nullptr, GENERIC_ALL, nullptr, &s->hIo)) || FAILED(d->CreateSharedHandle(s->sig, nullptr, GENERIC_ALL, nullptr, &s->hSig))) { logf("sync: CreateSharedHandle failed"); freeSync(s); return nullptr; }
    D3D12_DESCRIPTOR_RANGE rg[2] = {}; rg[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; rg[0].NumDescriptors = 1; rg[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV; rg[1].NumDescriptors = 1; rg[1].OffsetInDescriptorsFromTableStart = 1;
    D3D12_ROOT_PARAMETER rp[5] = {}; rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; rp[0].DescriptorTable.NumDescriptorRanges = 2; rp[0].DescriptorTable.pDescriptorRanges = rg;
    rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; rp[1].Descriptor.ShaderRegister = 1; rp[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; rp[2].Descriptor.ShaderRegister = 2;
    rp[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; rp[3].Constants.Num32BitValues = 14; rp[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; rp[4].Descriptor.ShaderRegister = 3;
    D3D12_ROOT_SIGNATURE_DESC rsd = {}; rsd.NumParameters = 5; rsd.pParameters = rp; ID3DBlob *sb = nullptr, *er = nullptr;
    if (FAILED(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sb, &er)) || FAILED(d->CreateRootSignature(0, sb->GetBufferPointer(), sb->GetBufferSize(), IID_PPV_ARGS(&s->rs)))) { logf("sync: root signature failed"); freeSync(s); return nullptr; }
    sb->Release();
    ID3D12PipelineState** pso[5] = {&s->imp, &s->spin, &s->exp, &s->cap, &s->expPk}; const char* ep[5] = {"imp", "spin", "exp", "cap", "exp"};
    const char* pk[4] = {"PACKED", "1", nullptr, nullptr};   // D3D_SHADER_MACRO[2], for expPk
    for (int i = 0; i < 5; i++) {
        ID3DBlob* cs = nullptr; er = nullptr;
        if (FAILED(comp(kSyncHlsl, strlen(kSyncHlsl), "nr_sync", i == 4 ? pk : nullptr, nullptr, ep[i], "cs_5_0", 1 << 15, 0, &cs, &er))) { logf("sync: shader %s: %s", ep[i], er ? (char*)er->GetBufferPointer() : "?"); freeSync(s); return nullptr; }
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {}; pd.pRootSignature = s->rs; pd.CS = {cs->GetBufferPointer(), cs->GetBufferSize()};
        HRESULT h = d->CreateComputePipelineState(&pd, IID_PPV_ARGS(pso[i])); cs->Release(); if (FAILED(h)) { logf("sync: PSO %s failed", ep[i]); freeSync(s); return nullptr; }
    }
    D3D12_DESCRIPTOR_HEAP_DESC hd = {}; hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors = 2 * kSyncRing; hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(d->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&s->heap)))) { logf("sync: descriptor heap failed"); freeSync(s); return nullptr; }
    s->incr = d->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    return s;
}

// Model resolution step: data_dir holds one subfolder per step (640x360, 960x540, 1280x720, ...), or is itself one
// data dir (legacy). nr_port.ini model_res=WxH picks a step; auto (default) takes the step nearest in height to the size
// the host creates the feature with (OptiScaler's Model resolution slider sets that size; a change recreates the feature).
static bool readMeta(const std::string& dir, unsigned* w, unsigned* h);
static std::string pickModelDir(unsigned w, unsigned h) {
    unsigned mw, mh; if (readMeta(g_dataDir, &mw, &mh)) return g_dataDir;
    std::vector<std::pair<unsigned, unsigned>> st; WIN32_FIND_DATAA fd; HANDLE fh = FindFirstFileA((g_dataDir + "\\*x*").c_str(), &fd);
    if (fh != INVALID_HANDLE_VALUE) { do { unsigned a, b; if ((fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && sscanf_s(fd.cFileName, "%ux%u", &a, &b) == 2) st.push_back({a, b}); } while (FindNextFileA(fh, &fd)); FindClose(fh); }
    auto name = [](unsigned a, unsigned b) { char n[32]; snprintf(n, sizeof n, "%ux%u", a, b); return std::string(n); };
    if (st.empty()) return g_dataDir;   // create reports the missing meta.txt
    std::string want = g_modelRes;
    if (want != "auto" && std::none_of(st.begin(), st.end(), [&](const std::pair<unsigned, unsigned>& p) { return name(p.first, p.second) == want; })) { logf("model_res %s: no such step in %s, using auto", want.c_str(), g_dataDir.c_str()); want = "auto"; }
    if (want == "auto" && !st.empty()) {
        auto best = std::make_pair(1280u, 720u); bool def = std::find(st.begin(), st.end(), best) != st.end(); if (!def) best = st[0];
        if (h) for (auto& p : st) if (abs((int)p.second - (int)h) < abs((int)best.second - (int)h)) best = p;   // ties keep 1280x720
        want = name(best.first, best.second);
    }
    std::string steps; for (auto& p : st) steps += " " + name(p.first, p.second);
    logf("model resolution: %s (model_res=%s, host size %ux%u, steps:%s)", want.c_str(), g_modelRes.c_str(), w, h, steps.c_str());
    return g_dataDir + "\\" + want;
}

// ---------------------------------------------------------------- exports
static bool readMeta(const std::string& dir, unsigned* w, unsigned* h) {
    std::string p = dir + "\\meta.txt"; FILE* f = nullptr; fopen_s(&f, p.c_str(), "r"); if (!f) return false;
    char line[256]; *w = *h = 0;
    while (fgets(line, sizeof line, f)) { unsigned v; if (sscanf_s(line, "width=%u", &v) == 1) *w = v; if (sscanf_s(line, "height=%u", &v) == 1) *h = v; }
    fclose(f); return *w && *h;
}

extern "C" {
__declspec(dllexport) int dlssnr_call_last_init = 1, dlssnr_call_last_create = 1;
__declspec(dllexport) int dlssnr_last_ratio_result = 0, dlssnr_last_ratio_stage = 0;
__declspec(dllexport) const char* dlssnr_backend_id() { return "rdna2-hip"; }   // presence marks the "port" flavour
__declspec(dllexport) const char* dlssnr_call_error() { return g_err; }
__declspec(dllexport) int dlssnr_port_state(void* f) { return f ? ((Feature*)f)->state.load() : -1; }   // diagnostics/tests: 0 loading, 1-3 async, 4 failed, 5 sync
__declspec(dllexport) void dlssnr_port_stats(void* f, unsigned* nr, unsigned* native, unsigned* reused) { Sync* s = f ? ((Feature*)f)->sy : nullptr; *nr = s ? s->ran : 0; *native = s ? s->busy : 0; *reused = s ? s->reused : 0; }   // tests: sync-path frame counts
__declspec(dllexport) void dlssnr_call_set_float_slot(int) {}
__declspec(dllexport) void dlssnr_call_probe_float(void*, const char*, float, int) {}
__declspec(dllexport) void dlssnr_call_set_extras(void*, float, ID3D12Resource*, ID3D12Resource*, ID3D12Resource*, unsigned, unsigned, unsigned, unsigned) {}

__declspec(dllexport) void* dlssnr_call_create(const wchar_t* snippet, const wchar_t*, ID3D12Device* dev, ID3D12GraphicsCommandList*, void*,
        unsigned w, unsigned h, int, float, int, float, float, float, int, int) {
    try {
        if (!g_log) loadSettings();
        logf("create: snippet=%ls dev=%p %ux%u", snippet ? snippet : L"", (void*)dev, w, h);
        dlssnr_call_last_create = 0;
        if (!dev) { seterr("create: no D3D12 device"); return nullptr; }
        unsigned mw = 0, mh = 0; const std::string dir = pickModelDir(w, h);
        if (!readMeta(dir, &mw, &mh)) { seterr("create: no meta.txt in '%s' (run gen_runtime_data.py, see INSTALL.md)", dir.c_str()); return nullptr; }
        if ((mw != w || mh != h) && g_sync) { logf("create: frame %ux%u, model %ux%u - sync path resamples (universal for any resolution / upscaler mode)", w, h, mw, mh); w = mw; h = mh; }
        if (mw != w || mh != h) { seterr("create: model size %ux%u but data dir '%s' is for %ux%u - generate data for this size", w, h, dir.c_str(), mw, mh); return nullptr; }
        auto* F = new Feature; F->dir = dir; F->W = w; F->H = h; F->pitch = align256(w * 8);
        F->tin.resize((size_t)w * h * 8); F->tout.resize((size_t)w * h * 8);
        UINT64 sz = (UINT64)F->pitch * h;
        F->rb = makeBuffer(dev, D3D12_HEAP_TYPE_READBACK, sz, &F->rbp);
        F->up[0] = makeBuffer(dev, D3D12_HEAP_TYPE_UPLOAD, sz, &F->upp[0]); F->up[1] = makeBuffer(dev, D3D12_HEAP_TYPE_UPLOAD, sz, &F->upp[1]);
        if (!F->rb || !F->up[0] || !F->up[1]) { seterr("create: cannot allocate staging buffers"); if (F->rb) F->rb->Release(); for (auto u : F->up) if (u) u->Release(); delete F; return nullptr; }
        F->dev = dev;
        if (g_apply) {
            F->upOld[0] = makeBuffer(dev, D3D12_HEAP_TYPE_UPLOAD, sz, &F->upOldp[0]); F->upOld[1] = makeBuffer(dev, D3D12_HEAP_TYPE_UPLOAD, sz, &F->upOldp[1]);
            F->zero = makeBuffer(dev, D3D12_HEAP_TYPE_UPLOAD, sz, &F->zerop);
            if (F->upOld[0] && F->upOld[1] && F->zero) { memset(F->zerop, 0, (size_t)sz); F->delta = makeDelta(F); } else logf("delta: cannot allocate extra buffers, falling back to copy");
        }
        if (g_sync) F->sy = makeSync(F);
        logf("create: sync %s; async fallback apply mode %s gain %.2f", F->sy ? "resources ready" : "off", F->delta ? "delta (live frame + NR edit)" : "copy (stale NR image)", g_gain);
        F->th = std::thread(worker, F);
        dlssnr_call_last_create = 1; return F;
    } catch (...) { seterr("create: exception"); return nullptr; }
}

__declspec(dllexport) int dlssnr_call_evaluate_v2(ID3D12GraphicsCommandList* cmd, void* f, void*, ID3D12Resource* color, ID3D12Resource*,
        ID3D12Resource*, ID3D12Resource* out, unsigned w, unsigned h, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
        int, int reset, float, int, float, float, float, int, float, float) {
    try {
        auto* F = (Feature*)f; int n = ++g_evalCalls;
        if (!F || !cmd || !color || !out) return 0;
        if (F->state == S_FAILED) return 0;
        if ((w != F->W || h != F->H) && F->state != S_SYNC) { if (!F->warnedSize) { F->warnedSize = true; logf("evaluate: frame %ux%u, model %ux%u - native frame until the runtime has loaded", w, h, F->W, F->H); } return 0; }
        D3D12_RESOURCE_DESC cd = color->GetDesc(), od = out->GetDesc();
        // R9G9B9E5 output (RE Engine): sync path only, packed in a shader and copied in. Only when the host created the texture
        // with ALLOW_UNORDERED_ACCESS: then it is in the UAV state (OptiScaler-DLSSNR contract) and our UAV<->COPY_DEST barriers
        // are legal. Without the flag a UAV-state barrier removes the device, so that case stays native.
        const bool e5 = F->state == S_SYNC && od.Format == DXGI_FORMAT_R9G9B9E5_SHAREDEXP && (od.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        if (F->state == S_SYNC && od.Format == DXGI_FORMAT_R9G9B9E5_SHAREDEXP && !e5) {
            if (!F->warnedFmt) { F->warnedFmt = true; logf("evaluate: R9G9B9E5 output without UAV flag (%llux%u) - native frame; try NR before super resolution", od.Width, od.Height); } return 0; }
        if (!(fmtOk(cd.Format) || (F->state == S_SYNC && cd.Format == DXGI_FORMAT_R9G9B9E5_SHAREDEXP)) || !(fmtOk(od.Format) || e5) || cd.Width < w || cd.Height < h || od.Width < w || od.Height < h) {
            if (!F->warnedFmt) { F->warnedFmt = true; logf("evaluate: unsupported color/output format %d/%d (%llux%u / %llux%u) - only RGBA16F / R11G11B10_FLOAT / RGBA8_UNORM", (int)cd.Format, (int)od.Format, cd.Width, cd.Height, od.Width, od.Height); } return 0; }
        F->inFmt = (int)cd.Format; F->outFmt = (int)od.Format;
        if (reset == 1) { F->published = -1; F->resetGen++; }      // a real reset (first frame, resolution change, camera cut): no edit from before it survives. Only exactly 1 counts.
        if (n <= 3) logf("evaluate #%d state %d published %d", n, F->state.load(), F->published.load());

        if (F->state == S_SYNC) {                                   // same-frame path: everything below runs on the GPU in the host's list order
            Sync* s = F->sy; ID3D12GraphicsCommandList2* c2 = nullptr;
            if (FAILED(cmd->QueryInterface(IID_PPV_ARGS(&c2)))) { if (!s->busy++) logf("sync: host list has no WriteBufferImmediate (ID3D12GraphicsCommandList2) - native frame"); return 0; }
            c2->Release();                                          // the host holds the list; we only need the interface for this call
            if (reset == 1) s->minSeq = s->seq + 1;                 // fg: no edit from before a camera cut is reused
            const unsigned seq = s->seq + 1; ++s->frm;
            int q = p_extEnqueue(F->ctx, seq, g_maxInflight);       // HIP: wait flag_in >= seq, run, flag_out = seq (queued now, runs when the list below executes)
            if (q < 0) { seterr("nr_ext_enqueue failed (%d)", q); F->state = S_FAILED; return 0; }
            if (q != 0 && !g_fg) { if (s->busy++ < 5 || s->busy % 100 == 0) logf("sync: frame not queued (HIP still busy with older frames, %u so far) - native frame", s->busy); return 0; }
            if (q == 0) { s->seq = seq; s->frmOf[seq % 16] = s->frm; if (++s->ran <= 3 || s->ran % 1000 == 0) logf("sync: NR frame %u queued (frame %ux%u, %u native, %u reused so far)", s->ran, w, h, s->busy, s->reused); }
            else s->reused++;                                       // fg: HIP busy -> live frame + last NR edit, the queue does not wait
            UINT k = (++s->rec % kSyncRing) * 2; auto cpu = s->heap->GetCPUDescriptorHandleForHeapStart(); auto gpu = s->heap->GetGPUDescriptorHandleForHeapStart();
            D3D12_CPU_DESCRIPTOR_HANDLE c0 = cpu, c1 = cpu; c0.ptr += (SIZE_T)k * s->incr; c1.ptr += (SIZE_T)(k + 1) * s->incr; gpu.ptr += (UINT64)k * s->incr;
            D3D12_SHADER_RESOURCE_VIEW_DESC sv = {}; sv.Format = cd.Format; sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D; sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; sv.Texture2D.MipLevels = 1;
            F->dev->CreateShaderResourceView(color, &sv, c0);
            D3D12_UNORDERED_ACCESS_VIEW_DESC uv = {}; uv.Format = e5 ? DXGI_FORMAT_R16G16B16A16_FLOAT : od.Format; uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D; F->dev->CreateUnorderedAccessView(e5 ? nullptr : out, nullptr, &uv, c1);   // e5: null UAV, output goes through ob
            UINT cb[14] = { F->W, F->H, seq, (UINT)(g_fg ? g_fgWaitMs : g_spinMs) * 6000u /* ~0.16 us per poll on a 6900 XT */, (UINT)s->outOff, (UINT)s->flagOff, 0, w | (h << 16), g_fg ? s->minSeq : 0u, 0, 0, (UINT)g_gate, s->frm, (UINT)g_fgMaxAge };
            float gain = g_gain; memcpy(&cb[6], &gain, 4); memcpy(&cb[9], &g_gateLo, 4); memcpy(&cb[10], &g_gateHi, 4);   // gLo, gHi
            ID3D12DescriptorHeap* hp[] = { s->heap }; cmd->SetDescriptorHeaps(1, hp); cmd->SetComputeRootSignature(s->rs);
            cmd->SetComputeRootDescriptorTable(0, gpu); cmd->SetComputeRootUnorderedAccessView(1, s->io->GetGPUVirtualAddress()); cmd->SetComputeRootUnorderedAccessView(2, s->st->GetGPUVirtualAddress());
            D3D12_RESOURCE_BARRIER uav = {}; uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;   // null resource = all UAV writes
            if (g_fg && s->seq > (q == 0 ? 1u : 0u)) {              // fg: harvest the newest earlier frame if HIP finished it after its own (short) wait
                UINT hb[14]; memcpy(hb, cb, sizeof hb); hb[2] = q == 0 ? seq - 1 : s->seq; hb[3] = 1; hb[12] = s->frmOf[hb[2] % 16]; cmd->SetComputeRoot32BitConstants(3, 14, hb, 0);
                cmd->SetPipelineState(s->spin); cmd->Dispatch(1, 1, 1); cmd->ResourceBarrier(1, &uav);
                cmd->SetPipelineState(s->cap); cmd->Dispatch((F->W + 7) / 8, (F->H + 7) / 8, 1); cmd->ResourceBarrier(1, &uav);
            }
            cmd->SetComputeRoot32BitConstants(3, 14, cb, 0);
            if (q != 0) {                                           // fg, HIP busy: st[0] = 0 forces the reuse/native branch of exp
                cb[3] = 0; cb[2] = s->seq + 1; cmd->SetComputeRoot32BitConstants(3, 14, cb, 0);   // spin with 0 polls: flag < seq+1 -> st[0] = 0
                cmd->SetPipelineState(s->spin); cmd->Dispatch(1, 1, 1); cmd->ResourceBarrier(1, &uav);
                cmd->SetPipelineState(s->exp); cmd->Dispatch((w + 7) / 8, (h + 7) / 8, 1); cmd->ResourceBarrier(1, &uav);
                return 1;
            }
            cmd->SetPipelineState(s->imp); cmd->Dispatch((F->W + 7) / 8, (F->H + 7) / 8, 1); cmd->ResourceBarrier(1, &uav);
            auto tr = [&](D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) { D3D12_RESOURCE_BARRIER t = {}; t.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; t.Transition.pResource = s->sig; t.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES; t.Transition.StateBefore = a; t.Transition.StateAfter = b; cmd->ResourceBarrier(1, &t); };
            tr(D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
            D3D12_WRITEBUFFERIMMEDIATE_PARAMETER wp = { s->sig->GetGPUVirtualAddress(), seq }; D3D12_WRITEBUFFERIMMEDIATE_MODE wm = D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT;
            c2->WriteBufferImmediate(1, &wp, &wm);                  // input is in io: release the HIP stream
            tr(D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
            cmd->SetPipelineState(s->spin); cmd->Dispatch(1, 1, 1); cmd->ResourceBarrier(1, &uav);   // wait (bounded) for the HIP stream's flag_out = seq
            if (g_fg) { cmd->SetPipelineState(s->cap); cmd->Dispatch((F->W + 7) / 8, (F->H + 7) / 8, 1); cmd->ResourceBarrier(1, &uav); }
            if (e5) {                                               // R9G9B9E5 has no typed UAV store: pack into a buffer, copy into the texture
                const UINT pitch = (w * 4 + 255) & ~255u;
                if (!s->ob || s->obW != w || s->obH != h) {
                    if (s->ob) s->ob->Release(); s->ob = nullptr; D3D12_HEAP_PROPERTIES hp = {D3D12_HEAP_TYPE_DEFAULT};
                    D3D12_RESOURCE_DESC bd = {}; bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = (UINT64)pitch * h; bd.Height = 1; bd.DepthOrArraySize = 1; bd.MipLevels = 1;
                    bd.SampleDesc.Count = 1; bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; bd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
                    HRESULT hr = F->dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&s->ob));
                    if (FAILED(hr)) { s->ob = nullptr; logf("sync: R9G9B9E5 staging buffer failed (0x%08lx) - native frame", (unsigned long)hr); return 0; }
                    s->obW = w; s->obH = h; logf("sync: output is R9G9B9E5 - packed in a shader and copied (%ux%u)", w, h);
                }
                cmd->SetComputeRootUnorderedAccessView(4, s->ob->GetGPUVirtualAddress());
                cmd->SetPipelineState(s->expPk); cmd->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
                D3D12_RESOURCE_BARRIER b[2] = {}; for (auto& x : b) { x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; x.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES; }
                b[0].Transition.pResource = s->ob; b[0].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; b[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
                b[1].Transition.pResource = out; b[1].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; b[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
                cmd->ResourceBarrier(2, b);
                D3D12_TEXTURE_COPY_LOCATION dl = {}, sl = {}; dl.pResource = out; dl.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                sl.pResource = s->ob; sl.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; sl.PlacedFootprint.Footprint = {DXGI_FORMAT_R9G9B9E5_SHAREDEXP, w, h, 1, pitch};
                D3D12_BOX bx = {0, 0, 0, w, h, 1}; cmd->CopyTextureRegion(&dl, 0, 0, 0, &sl, &bx);
                for (auto& x : b) std::swap(x.Transition.StateBefore, x.Transition.StateAfter);
                cmd->ResourceBarrier(2, b);
                return 1;
            }
            cmd->SetComputeRootUnorderedAccessView(4, s->st->GetGPUVirtualAddress());   // unused by exp; bound so the root signature is complete
            cmd->SetPipelineState(s->exp); cmd->Dispatch((w + 7) / 8, (h + 7) / 8, 1); cmd->ResourceBarrier(1, &uav);
            return 1;
        }

        D3D12_BOX box = {0, 0, 0, w, h, 1};
        auto barrier = [&](ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
            D3D12_RESOURCE_BARRIER rb = {}; rb.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; rb.Transition.pResource = r;
            rb.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES; rb.Transition.StateBefore = a; rb.Transition.StateAfter = b; cmd->ResourceBarrier(1, &rb); };
        auto footprint = [&](ID3D12Resource* buf, DXGI_FORMAT fmt) { D3D12_TEXTURE_COPY_LOCATION l = {}; l.pResource = buf; l.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            l.PlacedFootprint.Footprint.Format = fmt; l.PlacedFootprint.Footprint.Width = w; l.PlacedFootprint.Footprint.Height = h;
            l.PlacedFootprint.Footprint.Depth = 1; l.PlacedFootprint.Footprint.RowPitch = F->pitch; return l; };
        auto tex = [&](ID3D12Resource* t) { D3D12_TEXTURE_COPY_LOCATION l = {}; l.pResource = t; l.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; return l; };

        if (F->state == S_IDLE) {                                   // start a job: grab this frame
            memset(F->rbp, 0xFF, (size_t)F->pitch * F->H);          // sentinel; the worker sees it disappear when the copy lands
            barrier(color, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
            auto dst = footprint(F->rb, cd.Format), src = tex(color); cmd->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);
            barrier(color, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            F->jobGen = F->resetGen; F->jobStart = GetTickCount(); F->state = S_WAIT_COPY;
        }
        int slot = F->published; if (slot >= 0) F->slotEval[slot] = (unsigned)n;
        if (slot < 0 && !F->delta) return 0;   // copy mode: no answer yet, host keeps the native frame (delta mode passes the live frame through instead)
        if (F->delta) {                                             // out = current colour + stale NR edit, computed on the GPU in the host's list
            UINT k = (UINT)(n % kRing) * kDesc; auto cpu = F->heap->GetCPUDescriptorHandleForHeapStart(); auto gpu = F->heap->GetGPUDescriptorHandleForHeapStart();
            auto at = [&](UINT i) { D3D12_CPU_DESCRIPTOR_HANDLE h = cpu; h.ptr += (SIZE_T)(k + i) * F->incr; return h; };
            D3D12_SHADER_RESOURCE_VIEW_DESC sv = {}; sv.Format = cd.Format; sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D; sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; sv.Texture2D.MipLevels = 1;
            F->dev->CreateShaderResourceView(color, &sv, at(0));
            D3D12_SHADER_RESOURCE_VIEW_DESC bv = {}; bv.Format = DXGI_FORMAT_R32_TYPELESS; bv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER; bv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            bv.Buffer.NumElements = (UINT)((UINT64)F->pitch * F->H / 4); bv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            F->dev->CreateShaderResourceView(slot < 0 ? F->zero : F->up[slot], &bv, at(1)); F->dev->CreateShaderResourceView(slot < 0 ? F->zero : F->upOld[slot], &bv, at(2));
            D3D12_UNORDERED_ACCESS_VIEW_DESC uv = {}; uv.Format = od.Format; uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D; F->dev->CreateUnorderedAccessView(out, nullptr, &uv, at(3));
            ID3D12DescriptorHeap* hp[] = { F->heap }; cmd->SetDescriptorHeaps(1, hp); cmd->SetComputeRootSignature(F->rs); cmd->SetPipelineState(F->pso);
            gpu.ptr += (UINT64)k * F->incr; cmd->SetComputeRootDescriptorTable(0, gpu);
            UINT cb[6] = { w, h, F->pitch, (UINT)(g_gate && slot >= 0), 0, 0 }; memcpy(&cb[4], &g_gateLo, 4); memcpy(&cb[5], &g_gateHi, 4); cmd->SetComputeRoot32BitConstants(1, 6, cb, 0);
            cmd->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
            D3D12_RESOURCE_BARRIER ub = {}; ub.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; ub.UAV.pResource = out; cmd->ResourceBarrier(1, &ub);
            return 1;
        }
        barrier(out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
        auto dst = tex(out), src = footprint(F->up[slot], od.Format); cmd->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);
        barrier(out, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        return 1;
    } catch (...) { return 0; }
}

__declspec(dllexport) void dlssnr_call_release(void* f) {
    auto* F = (Feature*)f; if (!F) return;
    try {
        logf("release %p", f); F->quit = true;
        if (g_exiting || F->state == S_LOADING) { if (F->th.joinable()) F->th.detach(); return; }   // process exit, or the runtime is still loading: never block the game, leak F (the worker leaves on its own)
        if (F->th.joinable()) F->th.join();
        if (F->ctx && p_destroy) p_destroy(F->ctx);
        if (F->rb) F->rb->Release(); for (auto u : F->up) if (u) u->Release();
        for (auto u : F->upOld) if (u) u->Release(); if (F->zero) F->zero->Release();
        if (F->pso) F->pso->Release(); if (F->rs) F->rs->Release(); if (F->heap) F->heap->Release();
        freeSync(F->sy);   // ponytail: assumes the host released after its last list using us completed (INTERFACE.md: it parks handles); a still-spinning list ends by its bound
    } catch (...) {}
    delete F;
}
}

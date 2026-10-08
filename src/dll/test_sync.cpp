// test_sync <color.bin RGBA16F WxH> <dll dir> [frames]
// Plays the host for the sync path: one D3D12 direct queue; per frame ONE command list = upload this frame's colour ->
// evaluate_v2 -> copy `output` to readback (the consumer sits in the same list, after evaluate, as in the game).
// Every frame gets a different input (the base image shifted by 13 px per frame), so a stale or skipped result cannot match.
// Reference: nr_run from the same nr_runtime.dll in a second context fed the same frame sequence.
// Every GPU wait has a timeout (deadlock -> FAIL, never a hang). Env: NR_TEST_W/H (default 1280x720), NR_TEST_FMT=r11 (R11G11B10 textures).
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "dlssnr_iface.h"
#include "r11g10.h"
#define CK(x) do { HRESULT _h = (x); if (FAILED(_h)) { printf("FAIL %s -> 0x%08lx (line %d)\n", #x, (unsigned long)_h, __LINE__); return 10; } } while (0)
static UINT W = 1280, H = 720, MW, MH, UP = 1; static DXGI_FORMAT FMT = DXGI_FORMAT_R16G16B16A16_FLOAT; static UINT BPP = 8;   // W,H frame; MW,MH model (NR_TEST_UP=2: frame = model upsampled 2x, tests the resampling path)
static ID3D12Device* dev; static ID3D12CommandQueue* q; static ID3D12CommandAllocator* al; static ID3D12GraphicsCommandList* cl; static ID3D12Fence* fence; static UINT64 fv; static HANDLE ev;
static double now() { LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return c.QuadPart * 1000.0 / f.QuadPart; }
static bool wait(DWORD ms) { q->Signal(fence, ++fv); if (fence->GetCompletedValue() >= fv) return true; fence->SetEventOnCompletion(fv, ev); return WaitForSingleObject(ev, ms) == WAIT_OBJECT_0; }
static ID3D12Resource* buf(D3D12_HEAP_TYPE t, UINT64 sz, D3D12_RESOURCE_STATES st) {
    D3D12_HEAP_PROPERTIES hp = {}; hp.Type = t; D3D12_RESOURCE_DESC d = {}; d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; d.Width = sz; d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1;
    d.SampleDesc.Count = 1; d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; ID3D12Resource* r = nullptr; dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, st, nullptr, IID_PPV_ARGS(&r)); return r; }
static ID3D12Resource* tex(D3D12_RESOURCE_FLAGS fl, D3D12_RESOURCE_STATES st) {
    D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_DEFAULT; D3D12_RESOURCE_DESC d = {}; d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; d.Width = W; d.Height = H; d.DepthOrArraySize = 1;
    d.MipLevels = 1; d.Format = FMT; d.SampleDesc.Count = 1; d.Flags = fl; ID3D12Resource* r = nullptr; dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, st, nullptr, IID_PPV_ARGS(&r)); return r; }
static void bar(ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
    D3D12_RESOURCE_BARRIER x = {}; x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; x.Transition.pResource = r; x.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    x.Transition.StateBefore = a; x.Transition.StateAfter = b; cl->ResourceBarrier(1, &x); }
static D3D12_TEXTURE_COPY_LOCATION fp(ID3D12Resource* b, UINT pitch) { D3D12_TEXTURE_COPY_LOCATION l = {}; l.pResource = b; l.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; l.PlacedFootprint.Footprint = {FMT, W, H, 1, pitch}; return l; }
static D3D12_TEXTURE_COPY_LOCATION ts(ID3D12Resource* t) { D3D12_TEXTURE_COPY_LOCATION l = {}; l.pResource = t; l.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; return l; }
static double h2f(uint16_t h) { int e = (h >> 10) & 31, m = h & 1023; double v = e == 0 ? ldexp((double)m, -24) : e == 31 ? 1e9 : ldexp((double)(1024 + m), e - 25); return (h & 0x8000) ? -v : v; }
typedef void* (*P_create)(const char*, int, int, int, char*, int); typedef int (*P_run)(void*, const void*, void*, float*);

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (getenv("NR_TEST_W")) { W = atoi(getenv("NR_TEST_W")); H = atoi(getenv("NR_TEST_H")); }
    MW = W; MH = H; if (getenv("NR_TEST_UP")) { UP = atoi(getenv("NR_TEST_UP")); W = MW * UP; H = MH * UP; }
    const bool r11 = getenv("NR_TEST_FMT") && !strcmp(getenv("NR_TEST_FMT"), "r11"); if (r11) { FMT = DXGI_FORMAT_R11G11B10_FLOAT; BPP = 4; }
    if (argc < 3) { puts("usage: test_sync color.bin dll_dir [frames]"); return 1; }
    const int frames = argc > 3 ? atoi(argv[3]) : 300; std::string dir = argv[2]; SetDllDirectoryA(dir.c_str());
    std::vector<uint16_t> base((size_t)MW * MH * 4); { FILE* f = fopen(argv[1], "rb"); if (!f || fread(base.data(), 8, (size_t)MW * MH, f) != (size_t)MW * MH) { puts("cannot read color.bin"); return 1; } fclose(f); }
    char dd[1024]; GetPrivateProfileStringA("nr", "data_dir", "", dd, sizeof dd, (dir + "\\nr_port.ini").c_str());

    IDXGIFactory4* fac; CK(CreateDXGIFactory1(IID_PPV_ARGS(&fac))); IDXGIAdapter1* ad = nullptr;
    for (UINT i = 0;; i++) { IDXGIAdapter1* a; if (fac->EnumAdapters1(i, &a) != S_OK) break; DXGI_ADAPTER_DESC1 d; a->GetDesc1(&d); if (d.VendorId == 0x1002) { ad = a; break; } a->Release(); }
    if (!ad) { puts("no AMD adapter"); return 1; }
    CK(D3D12CreateDevice(ad, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev)));
    D3D12_COMMAND_QUEUE_DESC qd = {}; CK(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&q)));
    CK(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&al)));
    CK(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, al, nullptr, IID_PPV_ARGS(&cl))); cl->Close();
    CK(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))); ev = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    const UINT pitch = (W * BPP + 255) & ~255u; const UINT64 sz = (UINT64)pitch * H;
    auto color = tex(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE), out = tex(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    auto up = buf(D3D12_HEAP_TYPE_UPLOAD, sz, D3D12_RESOURCE_STATE_GENERIC_READ), rb = buf(D3D12_HEAP_TYPE_READBACK, sz, D3D12_RESOURCE_STATE_COPY_DEST);
    uint8_t *upp, *rbp; up->Map(0, nullptr, (void**)&upp); rb->Map(0, nullptr, (void**)&rbp);

    HMODULE m = LoadLibraryA((dir + "\\nvngx.dll_dlssnr.dll").c_str()); if (!m) { printf("load failed %lu\n", GetLastError()); return 1; }
    auto create = (PFN_NrCreate)GetProcAddress(m, "dlssnr_call_create"); auto eval = (PFN_NrEvaluate)GetProcAddress(m, "dlssnr_call_evaluate_v2");
    auto release = (PFN_NrRelease)GetProcAddress(m, "dlssnr_call_release"); auto err = (PFN_NrError)GetProcAddress(m, "dlssnr_call_error");
    auto state = (int (*)(void*))GetProcAddress(m, "dlssnr_port_state");
    HMODULE rt = GetModuleHandleA("nr_runtime.dll"); if (!rt) rt = LoadLibraryA((dir + "\\nr_runtime.dll").c_str());
    auto rcreate = (P_create)GetProcAddress(rt, "nr_create"); auto rrun = (P_run)GetProcAddress(rt, "nr_run");
    if (!create || !eval || !release || !state || !rcreate || !rrun) { puts("missing exports"); return 1; }

    al->Reset(); cl->Reset(al, nullptr); void* f = create(L"nvngx_dlssnr.dll", L"data", dev, cl, nullptr, W, H, 0, 1.f, 0, 1.f, 1.f, 1.f, 0, 1); cl->Close();
    { ID3D12CommandList* l[] = {cl}; q->ExecuteCommandLists(1, l); } if (!wait(5000)) { puts("FAIL timeout"); return 2; }
    if (!f) { printf("FAIL create: %s\n", err()); return 3; }
    char e2[512] = ""; void* ref = rcreate(dd, (int)MW, (int)MH, 1, e2, sizeof e2); if (!ref) { printf("FAIL reference nr_create: %s\n", e2); return 3; }
    double t0 = now(); while (state(f) == 0 && now() - t0 < 90000) Sleep(50);
    printf("dll state %d after %.0f ms (5 = sync)\n", state(f), now() - t0); if (state(f) != 5) { printf("FAIL sync path not active: %s\n", err()); return 4; }

    std::vector<uint16_t> in((size_t)MW * MH * 4), refo((size_t)MW * MH * 4);
    auto makeFrame = [&](int k) {   // shifted base image -> texture format bytes in upload, and the exact half4 the network sees in `in`
        int sh = (k * 13) % MW;
        for (UINT y = 0; y < H; y++) for (UINT x = 0; x < W; x++) {
            const uint16_t* s = &base[((size_t)(y / UP) * MW + (x / UP + sh) % MW) * 4]; uint16_t* d = &in[((size_t)(y / UP) * MW + x / UP) * 4];
            if (r11) { uint32_t v; pack_r11g11b10(s, &v); memcpy(upp + (size_t)y * pitch + x * 4, &v, 4); unpack_r11g11b10(v, d); }
            else { memcpy(d, s, 8); memcpy(upp + (size_t)y * pitch + x * 8, s, 8); } } };
    auto frame = [&](int k, bool exec, double* ms) -> int {   // one game frame; returns evaluate's result, -100 on GPU timeout
        al->Reset(); cl->Reset(al, nullptr);
        bar(color, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST); { auto d = ts(color), s = fp(up, pitch); cl->CopyTextureRegion(&d, 0, 0, 0, &s, nullptr); }
        bar(color, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        int r = k < 0 ? 1 : eval(cl, f, nullptr, color, nullptr, nullptr, out, W, H, W, H, W, H, 0, 0, 0, 0, 0, k == 0, 1.f, 0, 1.f, 1.f, 1.f, 0, 1.f, 1.f);
        bar(out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE); { auto d = fp(rb, pitch), s = ts(out); cl->CopyTextureRegion(&d, 0, 0, 0, &s, nullptr); }
        bar(out, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS); cl->Close();
        if (!exec) return r;                                  // the host dropped this list
        double a = now(); ID3D12CommandList* l[] = {cl}; q->ExecuteCommandLists(1, l); if (!wait(5000)) return -100; *ms = now() - a; return r; };
    double se_o = 0, se_i = 0; auto psnr = [&](double se) { return 10 * log10(3.0 * MW * MH / std::max(se, 1e-12)); };
    auto compare = [&](size_t* beyond, double* maxerr) {
        *beyond = 0; *maxerr = 0; se_o = se_i = 0;
        for (UINT y = 0; y < MH; y++) for (UINT x = 0; x < MW; x++) {
            double ov[3] = {0, 0, 0};   // UP > 1: mean of the UPxUP frame block (bilinear up + box down ~ identity)
            for (UINT j = 0; j < UP * UP; j++) { size_t fy = y * UP + j / UP, fx = x * UP + j % UP; uint16_t o[4];
                if (r11) unpack_r11g11b10(*(uint32_t*)(rbp + fy * pitch + fx * 4), o); else memcpy(o, rbp + fy * pitch + fx * 8, 8);
                for (int c = 0; c < 3; c++) ov[c] += h2f(o[c]) / (UP * UP); }
            for (int c = 0; c < 3; c++) {
                uint16_t b = refo[((size_t)y * MW + x) * 4 + c]; double bv = (b & 0x7C00) == 0x7C00 ? 0 : std::max(0.0, h2f(b));
                double e = fabs(ov[c] - bv);
                { double iv = h2f(in[((size_t)y * MW + x) * 4 + c]); se_o += std::min(e * e, 1.0); se_i += std::min((iv - bv) * (iv - bv), 1.0); }   // UP > 1 metric: PSNR (peak 1, clamped) of NR result vs reference, against doing nothing
                if (e > *maxerr) *maxerr = e; if (e > 0.004 + (r11 ? 0.035 : 0.004) * fabs(bv)) (*beyond)++; } } };   // r11: 5-6 bit mantissa

    // baseline: the same list without evaluate
    std::vector<double> tb, ts_, tr; double ms = 0;
    makeFrame(0); for (int i = 0; i < 20; i++) { if (frame(-1, true, &ms) == -100) { puts("FAIL timeout baseline"); return 5; } tb.push_back(ms); }
    int failed = 0, notRun = 0;
    for (int k = 0; k < frames; k++) {
        makeFrame(k);
        double a = now(); float gms = 0; if (rrun(ref, in.data(), refo.data(), &gms)) { puts("FAIL reference nr_run"); return 6; } tr.push_back(now() - a);
        int r = frame(k, true, &ms);
        if (r == -100) { printf("FAIL frame %d: GPU wait timed out (deadlock?)\n", k); return 7; }
        if (r != 1) { notRun++; printf("frame %d: evaluate returned %d\n", k, r); continue; }
        ts_.push_back(ms); size_t bad; double mx; compare(&bad, &mx);
        bool ok = UP > 1 ? psnr(se_o) >= 30 : bad <= (size_t)MW * MH * 3 / 1000; failed += !ok;   // UP > 1 (resampled): compare quality, not values
        if (k < 3 || k % 50 == 0 || !ok) printf("frame %d: %.1f ms list (ref nr_run %.1f ms) | %zu rgb values beyond tol, max err %.4g %s\n", k, ms, tr.back(), bad, mx, ok ? "ok" : "MISMATCH");
        if (UP > 1 && (k < 3 || !ok)) printf("  psnr vs reference: NR %.1f dB, no NR %.1f dB\n", psnr(se_o), psnr(se_i));
    }
    // robustness: the host records a frame but never executes it, then carries on
    makeFrame(frames); frame(frames, false, &ms); int after = 0, afterBad = 0; double afterMax = 0;
    for (int k = frames + 1; k < frames + 11; k++) { makeFrame(k); int r = frame(k, true, &ms); if (r == -100) { puts("FAIL deadlock after a dropped list"); return 8; } if (r == 1) after++; afterMax = std::max(afterMax, ms); afterBad += ms > 1000; }
    double tr0 = now(); release(f); double trel = now() - tr0;
    auto stat = [](std::vector<double> v, const char* n) { if (v.empty()) return; std::sort(v.begin(), v.end()); double s = 0; for (double x : v) s += x; printf("%-34s avg %6.2f ms  p50 %6.2f  p95 %6.2f  max %6.2f  (n=%zu)\n", n, s / v.size(), v[v.size() / 2], v[v.size() * 95 / 100], v.back(), v.size()); };
    stat(tb, "list without NR (baseline)"); stat(ts_, "list with sync NR (submit->done)"); stat(tr, "reference nr_run (CPU in/out)");
    printf("frames %d: %d evaluated+matched, %d mismatched, %d not run | after dropped list: %d/10 ran, worst %.1f ms | release %.0f ms\n", frames, (int)ts_.size() - failed, failed, notRun, after, afterMax, trel);
    bool pass = failed == 0 && notRun == 0 && after == 10 && afterBad == 0 && trel < 3000;
    puts(pass ? "PASS" : "FAIL"); fflush(stdout); return pass ? 0 : 9;
}

// Persistent in-process runtime for the RDNA2 DLSS-NR port: load once, run frames on demand.
// Data dir = output of gen_runtime_data.py (manifest.txt, kernargs.bin, arena.bin, meta.txt) for one resolution.
#pragma once
#include <stddef.h>
#ifdef _WIN32
#  ifdef NR_BUILD
#    define NR_API extern "C" __declspec(dllexport)
#  else
#    define NR_API extern "C" __declspec(dllimport)
#  endif
#else
#  define NR_API extern "C"
#endif
typedef struct nr_ctx nr_ctx;

// vit_every: 1 = compute every frame (default); 2 = skip the ViT dispatches on odd frames and reuse the previous
// result (needs data generated with VIT_OWN=1, else create fails). width/height must match the data dir.
// Returns NULL on failure and writes a message into err.
NR_API nr_ctx* nr_create(const char* data_dir, int width, int height, int vit_every, char* err, int errlen);
// in/out are width*height*8 bytes, RGBA16F, tightly packed (nr_bytes gives the size). Recurrent state persists
// between calls. gpu_ms may be NULL. Returns 0 on success.
NR_API int nr_run(nr_ctx*, const void* in, void* out, float* gpu_ms);
NR_API size_t nr_bytes(nr_ctx*);
// Restore the pristine arena (first-frame state): re-reads arena.bin from disk, takes seconds.
NR_API int nr_reset(nr_ctx*);
// Effect strength, adjustable at runtime (no new data dir). The network's strength/conditioning lives in two f32 pairs of the
// pre-block kernarg: (tone, structure) = PreParams +0x28 (data dirs made with the default harness: 0, 0) and (a, b) = +0x48 (default 1, 1).
// Known-good points (effect vs default, clean picture): (0,0,1,1) = default; (2,2,4,4) ~2.1x; (4,4,8,8) ~3.1x. Much larger values darken or
// shift brightness (see own/STRENGTH_RUNTIME.md). Values are 0..64 (NaN/out of range -> returns 2 and changes nothing). Thread-safe: takes
// effect at the start of the next nr_run. nr_get_strength returns the values in effect (the data dir's own until the first set).
NR_API int nr_set_strength(nr_ctx*, float tone, float structure, float a, float b);
NR_API int nr_get_strength(nr_ctx*, float* tone, float* structure, float* a, float* b);
// One-knob version: intensity 0..2 (clamped). 0 = (0,0,0,0) effect nearly off; 0..1 scales (a,b) from 0 to 1; 1 = default (0,0,1,1);
// 1..2 moves linearly to (4,4,8,8) = the largest clean look (~3x the effect, +37% local contrast). Returns 2 for NaN/negative.
NR_API int nr_set_intensity(nr_ctx*, float intensity);
NR_API void nr_destroy(nr_ctx*);

// ---- D3D12 interop: GPU-side per-frame sync, no host copies (see handover/game-ready-sync/DESIGN.md).
// io/sig = NT handles from ID3D12Device::CreateSharedHandle of two committed D3D12_HEAP_FLAG_SHARED buffers;
// *_alloc = GetResourceAllocationInfo().SizeInBytes, *_size = buffer width.
// io: model input RGBA16F packed (nr_bytes) at 0, model output at out_off, u32 flag_out at flag_off (runtime writes seq when done).
// sig: u32 flag_in at 0 (D3D12 writes seq once the input is in io, e.g. WriteBufferImmediate MARKER_OUT).
// Returns 0 on success; on failure writes err and leaves the context usable with nr_run.
NR_API int nr_ext_open(nr_ctx*, void* io, unsigned long long io_alloc, unsigned long long io_size, unsigned long long out_off,
                       unsigned long long flag_off, void* sig, unsigned long long sig_alloc, unsigned long long sig_size, char* err, int errlen);
// Queues frame `seq` (1, 2, 3, ...) on the runtime's own stream and returns at once: CPU thread polls flag_in >= seq (<= 500 ms, else frame skipped), input -> network,
// run, network -> output, flag_out = seq. 0 = queued, 1 = busy (max_inflight frames still pending), <0 = error.
NR_API int nr_ext_enqueue(nr_ctx*, unsigned seq, int max_inflight);
// Releases the stream from any pending wait (flag_in = 0xFFFFFFFF), waits <= timeout_ms for it to drain, frees the imports.
// Called by nr_destroy too. Returns 0 when the stream drained.
NR_API int nr_ext_close(nr_ctx*, int timeout_ms);   // timeout unused: the poll thread is bounded by its own 500 ms wait

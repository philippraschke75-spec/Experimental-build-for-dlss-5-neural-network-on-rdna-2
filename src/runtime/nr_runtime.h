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
// result (needs data generated with VIT_OWN=1, else create fails).
// vit_every = -N (N >= 2): reuse the whole low-res bottleneck (c512 encoder + ViT + c512 decoder) on N-1 of N frames. width/height must match the data dir.
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
// NVIDIA's conditioning (style 0..2, local tone / structure 0..2, skin -1 = follow structure, auto mask); intensity is the host's output blend.
NR_API int nr_set_conditioning(nr_ctx*, int style, float tone, float structure, float skin, int auto_mask);
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

// ---- Temporal history (ABI v2, additive: nr_run / nr_ext_enqueue keep history off and stay bit-exact).
// History = the previous frame's network output in proxy space P (post-block output rgb, values [0,1]) as RGB32F over the PADDED
// frame: pad_w x pad_h pixels, 12 B/px, row stride pad_w*12, image at the top-left (W x H), same indexing as k_import's RGB.
// hist_valid != 0 points pre-block +0x40 (input lanes 7-9) and post-block +0x40 (lerp toward history, weight gate*sigmoid(head.a)) at it;
// hist_valid = 2: post-block lerp only (pre-block +0x40 stays null) - stable on static frames, use this for now; 3 = pre-block only (debug).
// 0 = no history (first frame, reset, cut, resize, skipped/late frame). k_export +0x38 stays 0.
NR_API int nr_hist_layout(nr_ctx*, unsigned* pad_w, unsigned* pad_h);
// nr_run with history: the history is the previous nr_run2 output, unwarped (static camera / testing). Returns 5 on a history alloc/copy error.
NR_API int nr_run2(nr_ctx*, const void* in, void* out, float* gpu_ms, int hist_valid);
// Optional io slots in the nr_ext io buffer (offsets in bytes, 4-aligned, each pad_w*pad_h*12 bytes, not overlapping the input; 0 = unused):
//   hist_in_off:  the DLL writes the WARPED previous P here before signalling flag_in; used for frames enqueued with hist_valid=1.
//                 If 0, the runtime uses its own unwarped previous output instead.
//   hist_out_off: after every frame the runtime writes this frame's P here (before flag_out = seq), for the DLL's next warp.
// Call after nr_ext_open. Returns 0 ok, 1 not open, 2 bad offsets.
NR_API int nr_ext_history(nr_ctx*, unsigned long long hist_in_off, unsigned long long hist_out_off);
NR_API int nr_set_seed_mode(nr_ctx*, int mode);   // 1 = noise seed counts frames on history frames (0 on no-history frames, as NVIDIA); 0 = seed 0 always (default)
NR_API int nr_ext_enqueue2(nr_ctx*, unsigned seq, int max_inflight, int hist_valid);
// Split jobs (ABI v3, additive): part 1 = dispatches [0, k) after input + history, part 2 = [k, n) + history out + output; flag_out = seq after each
// part. Part 2 waits for its own flag_in too, so the D3D12 side places each half in a frame of its own. part 0 = whole frame (= enqueue2).
// nr_ext_split must be called first (once, before any enqueue): *k_full / *k_skip > 0 fix the split point, <= 0 measure it (half the GPU time;
// k_skip for frames that skip dispatches with vit_every > 1). ms (nullable, 4 floats): measured part 1 / part 2 ms for full and skipping frames.
NR_API int nr_ext_split(nr_ctx*, int* k_full, int* k_skip, float* ms);
NR_API int nr_ext_enqueue3(nr_ctx*, unsigned seq, int max_inflight, int hist_valid, int part);
// N-way split (additive): np parts (2..8); k_full / k_skip: np-1 split points each (> 0 fixed, <= 0 measured: ~1/np of the GPU time per part),
// ms (nullable, 2*np floats): measured part times, full frames then skipping frames. enqueue3 part 1..np: part 1 = input + history + [0, k1),
// part np = [k(np-1), n) + history out + output. nr_ext_split = nr_ext_splitn(np = 2).
NR_API int nr_ext_splitn(nr_ctx*, int np, int* k_full, int* k_skip, float* ms);
// Debug only: run the first n dispatches of each frame (0 = all); read arena bytes (offsets as in meta.json placements).
NR_API int nr_dbg_steps(nr_ctx*, int n);
NR_API int nr_dbg_read(nr_ctx*, unsigned long long off, unsigned long long size, void* dst);
// Log sink for the hardware line at create and the output self-check (default stderr). A failed self-check makes nr_ext_enqueue* return -2.
NR_API void nr_set_log(void (*cb)(const char*));
// Optional, before nr_create: luid = the host's D3D12 adapter LUID (8 bytes, ID3D12Device::GetAdapterLuid) or NULL; allow_any = 1 runs on
// GPUs other than RDNA2 dGPUs (gfx1030-1032). Without it nr_create picks the dGPU with the most CUs.
NR_API void nr_select_device(const void* luid, int allow_any);
// ---- additive (AMDNR lmxxf-ABI runtime, amdnr/rdna2_nr_runtime.cpp)
// Strict device check, nothing created: the HIP device with exactly this LUID (no fallback), an RDNA2 dGPU unless allow_any. Index or -1 + err.
NR_API int nr_check_device(const void* luid, int allow_any, char* err, int errlen);
// Explicit nvngx_dlssnr.dll for the weights (tried before the data dir's parents); NULL/"" = parents only. Before nr_create.
NR_API void nr_set_weights_dll(const char* path);
// Ext path progress without waiting: done_seq = last seq whose output reached io (flag_out written), proc_seq = last seq the ext thread
// finished with (ran, or dropped after its flag_in timeout), inflight = queued + running. Any pointer may be NULL.
NR_API int nr_ext_status(nr_ctx*, unsigned* done_seq, unsigned* proc_seq, int* inflight);
// Extra passes: after its network, c runs `next` (and next's own next, ...) on its stream, each fed the previous pass's output, no history;
// the last pass's output goes to io. next: same size and device, never nr_ext_open'ed. NULL unchains. Destroy c before its chain.
NR_API int nr_ext_chain(nr_ctx* c, nr_ctx* next);

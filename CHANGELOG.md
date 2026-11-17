# Changelog (HexNeural, formerly the DLSS 5 NR port for AMD RDNA2)

## v1.0.0 HexNeural (2026-11-17)

**Our biggest release, and no longer experimental.** The project is now **HexNeural**: DLSS 5 Neural Rendering, DLSS Super Resolution Model K / M and DLSS 4 Frame Generation on AMD RX 6000 (RDNA2), in one package, tested together in game.

> **Coming to OpenNR:** our RDNA2 runtime is being added to OpenNR within the next few hours.
> **AMDNR** (the community project, not AMD) plans to ship our port with its **v3.5.6** release.

### In game (Cyberpunk 2077 benchmark, RX 6900 XT, 1440p Quality, NR on, stock clocks)

| Setup | Real fps | Displayed fps |
|---|---|---|
| FSR 4.1 + NR | 59.5 | 59.5 |
| DLSS Model K + NR | 56.1 | 56.1 |
| DLSS Model K + NR + DLSS 4 FG 2x | 42.0 | **83.8** |
| DLSS Model K + NR + DLSS 4 FG 6x | 31.6 | **180.1** |

Displayed fps recorded with PresentMon (the game's counter only sees real frames).

### Changes since v0.5.2

**New: DLSS 4 Frame Generation (2x up to 6x)**
- NVIDIA's DLSS-G networks (weights read from your own `nvngx_dlssg.dll` 310.9.1) on our own RDNA2 runtime: both networks (DL1 once per real frame, DL2 per generated frame) and our own implementation of the frame generation pipeline.
- **Multi-frame generation 3x-6x** with our own pacing on FSR-FG's swapchain (FFX itself presents only one generated frame).
- Matches NVIDIA's generated frames (RTX 5070 Ti capture, same game frames) at **63-66 dB**.
- Offline, 1440p: DL1 0.73 ms, DL2 0.61 ms, rest of the pipeline ~1.17 ms. In game at 6x: 9.7 ms for 5 generated frames.
- Bounded waits, falls back to the real frame, switches itself off after repeated failures or when it would cost more than it gives.

**New: DLSS Super Resolution Model K and Model M**
- NVIDIA's transformer upscaler models (converted once from your own `nvngx_dlss.dll` 310.5+) on a WrongGPU-based runtime with our RDNA2 kernels.
- **Model K: 2.36 ms** at 1440p Quality (1706x960 -> 2560x1440, offline). In game about 1 ms per frame more than FSR 4.1.
- Model M: ~11.6 ms, labelled "slow on RDNA2" in the menu.
- Shown as "DLSS Model K" / "DLSS Model M" in the overlay's Upscaler list.

**DLSS 5 Neural Rendering**
- **New host: OpenNR** (our OptiScaler fork). Our runtime is the `NrBackend=rdna2` backend (`Rdna2NrRuntime.dll`). Replaces the v0.5.2 bridge (`nvngx.dll_dlssnr.dll`, `nr_port.ini`).
- **Faster network:** 23.5 -> **20.4 ms** at 1440p native with full ViT every frame; 10.8 ms at 960p. int8 paths for ViT and pre/post on by default; ViT runs every 2nd frame by default (`vit_every=2`).
- The network reads its inputs and writes output and history directly in the shared buffer: 3 copies less per frame.
- Fix: ~220 ms stalls when a job was turned down.
- Fix: NR never started when OptiScaler spoofed the GPU (an RX 6800 reported as "RTX 5090"). The GPU is now identified through HIP.
- **RX 6500 XT / 6400 (gfx1034) supported.** All RDNA2 dGPUs: gfx1030, 1031, 1032, 1034.
- Split network (`nr_split`) is off and greyed out in the menu: it trailed in motion.

**Everything together**
- Fix: NR + Frame Generation + Model K stalled (~380 ms per frame) because they share one HIP runtime and HIP's default 4 hardware queues deadlocked. OptiScaler now sets `GPU_MAX_HW_QUEUES=16` before any HIP user loads.

**Overlay (Insert)**
- Upscaler list with DLSS Model K / M; "RDNA2 network" section (ViT skip, network resolution, Hybrid/Classic); FG output "RDNA2 DLSS-FG" with a live status line and the **RDNA2 FG multiplier** (2x-6x, applies at once); menu in English.

**Package**
- New layout, no installer: copy the files into the game folder, add your own NVIDIA DLLs. `SHA256SUMS.txt` for every file. No NVIDIA files inside.
- Coming from v0.5.2: run its `Uninstall_DLSS5.bat` first.

### Offline network times (RX 6900 XT)

| Network | Resolution | Time |
|---|---|---|
| DLSS 5 NR, full ViT every frame | 2560x1440 | 20.4 ms |
| DLSS 5 NR | 1707x960 | 10.8 ms |
| DLSS SR Model K | 1706x960 -> 2560x1440 | 2.36 ms |
| DLSS SR Model M | 1706x960 -> 2560x1440 | ~11.6 ms |
| DLSS 4 FG DL1 / DL2 | 2560x1440 | 0.73 / 0.61 ms |

### Source
- OptiScaler fork: https://github.com/philippraschke75-spec/OpenNR-OptiScaler/tree/release
- WrongGPU (MIT): https://github.com/phantoslayer21/wronggpu. Parts based on WrongGPU (github.com/phantoslayer21/wronggpu), MIT.


## v0.5.2 (2026-11-09, experimental pre-release)

Faster network and much less "white jumping" grain. Same model (NVIDIA's new small model, 310.9.16).

**Faster network (~14 % at 1440p)**
- The FP8 (e4m3) emulation of the activations is now a 2-instruction rounding on f16 bits (3-bit mantissa, ±448 clamp only where needed) instead of the exact emulation; shorter e4m3 byte conversions; per-kernel occupancy and compiler mode tuned; built with ROCm 6.4.
- Not byte-identical to v0.5.1 any more: 46.8 dB to v0.5.1, distance to NVIDIA's output unchanged (33.3 vs 33.5 dB on our test frame), effect strength unchanged.
- Network time, RX 6900 XT:

| Output | v0.5.1 | v0.5.2 |
|---|---|---|
| 1080p | 16.3 ms | 14.4 ms (-11.5 %) |
| 1440p | 27.4 ms | 23.5 ms (-14 %) |
| 4K | 58.0 ms | 50.5 ms (-13 %) |

In game this is roughly +7-10 % FPS, depending on how much the game itself costs.

**Less grain: `split_fade=5` (was 3)**
- A new network result now fades in over 5 frames. Test (1440p, test_sync): white jumping pixels on a still camera 999 -> 291 (-71 %), on a pan 2809 -> 420 (-85 %), effect -1 %.
- More fade can mean a little more trailing on very fast motion; `split_fade=4` or `3` brings back the old behaviour.

**Source (src/)**
- `Q8FAST` / `Q8BYTE` build options in all rounding helpers; `src/native/build.sh` now builds exactly the shipped v0.5.2 kernels (CU/WGP mode and fast-math per module, occupancy per kernel).
- Public sources synced with the shipped kernels (`swin_var.hip`, `c512.hip`, `io_head.hip`); added the missing `wstage.h`; `io_head.hip` uses its own `nk_common_io.h`.
- Removed `images/ingame1.jpg` / `ingame2.jpg`: they showed an RTGI mod, not DLSS 5.

**Known issue**
- One report (Cyberpunk 2077) of a light, regular micro-stutter every few seconds; cause not found yet (our log is clean, no skipped or native frames).

## v0.5.1 (2026-10-22, experimental pre-release)

Same look as v0.5 (the network output is byte-identical), a bit faster, plus a grain filter and a new experimental pipeline mode. The default settings stay safe; the pipeline is opt-in.

**Faster network (output byte-identical to v0.5)**
- The ViT's global attention now runs on `v_dot2` (packed f16 dot products) instead of scalar f32 multiply-adds: 1.37-1.52 ms -> 0.81-0.83 ms at 1440p. All other kernels already used `v_dot2`/`v_dot4`.
- Network time, RX 6900 XT, hybrid mode:

| Output | v0.5 | v0.5.1 |
|---|---|---|
| 1080p | 17.0 ms | 16.8 ms |
| 1440p | 28.1 ms | 27.5 ms |
| 4K | 60.6 ms | 58.1 ms |

**Less static grain (new `edit_temporal_still=0.85`, on by default)**
- The network turns tiny frame-to-frame changes of the game image (TAA/FSR jitter, ray-tracing noise) into slightly different fine detail every frame. On a still camera this looked like static noise.
- Where the image stands still and did not change, the edit is now blended more strongly with the previous one. Test (1440p, still camera): white "jumping" pixels -46 %; on motion unchanged.
- Protected against ghosts: the stronger blend only applies where the pixel does not move, the game image is unchanged and the old and new edit agree (FO4's depth range made the depth test alone useless).
- `split_fade` now uses a coarse image-change test too, so a character that moves on no longer leaves its old edit behind as a shadow in games like FO4.
- `edit_temporal_still=0` = v0.5 behaviour.

**New, EXPERIMENTAL: pipeline mode (`pipeline=1`, preset `Hybrid_1440_Pipeline`)**
- The network runs beside the game instead of inside each frame and uses any idle GPU time; no frame waits for it.
- Simulated load, 1440p hybrid: CPU-bound game (FO4-like) 53 -> 72 FPS (nr_split 3 vs pipeline), GPU-bound (Cyberpunk-like) 31 -> 36 FPS on a still camera. In-game test FO4: up to ~50 FPS.
- On fast camera moves the network gets 1.5x more time per frame (`pipeline_motion`, `pipeline_motion_slice`) so the shown edit stays about as young as with nr_split 3.
- Known issues: more trailing on fast motion than nr_split, and in some games more grain. That is why it is off by default.

**Prepared: DLSS 5 before the upscaler**
- New hybrid steps 1707x960 and 1280x720 in `nr_data_h15` (13.8 ms / 9.7 ms). With OptiScaler's `RunBeforeSR=true` and FSR Quality the network runs at render resolution: about half the network time; on a still image ~93 % of the fine detail of the normal path. Not yet tested in motion.

**Other**
- Fixed a build setting that made a fresh build of the c64 kernel 30 % slower than the shipped one (shipped binaries were not affected).

**Upgrade:** copy `nvngx.dll_dlssnr.dll`, `nr_runtime.dll`, `nr_data_h15` and `nr_data` over v0.5, then pick a preset. Your own `nr_port.ini` keeps working; add `edit_temporal_still=0.85` for the grain filter.

## v0.5 (2026-10-20)

**The DLSS 5 look is finally here.** Every earlier version ran the network at 720p or 540p, so its detail blocks (skin pores, cloth, fine contrast) worked at a quarter of NVIDIA's pixels and the effect looked weak. v0.5 runs them at your output resolution.

**Full look (new defaults)**
- `data_dir=nr_data_h15`, `model_res=auto`: **hybrid mode**. The detail blocks run at your output size (1440p screen: 2560x1440), the main network 1.5x smaller. Near-full DLSS 5 look for about 60 % of the full cost (1440p: 29.7 ms instead of 48 ms network time on an RX 6900 XT).
- `model_res=auto` now picks the largest step not above your output (it was capped at 720p).
- In Cyberpunk at 1440p the on/off difference is about 3x larger than with the old 720p default.

**Ghosting and trailing fixed**
- **Motion vectors in games with an upscaler:** the vectors come at render resolution (e.g. 1707x960 with FSR Quality). They were treated as output pixels, so everything that is moved along them (history, edit stabilisation, nr_split) moved only 2/3 of the way and trailed. Now scaled correctly; `mv_scale=1` again.
- **ViT reuse (`vit_every=0`) on fast motion:** the reused middle stage was stale and trailed. New `vit_motion_gate=1`: on fast camera moves that frame is computed in full.

**nr_split is clean now (more FPS)**
- Default `nr_split=2` with new `split_fade=3`: a new network result fades in over 3 frames instead of popping. In the test the visible flicker during a camera pan drops from 9.5 % (no split) to 1.2 %.
- Halo fix (`halo_fix=1`): no bright halos/ghost copies around thin objects with nr_split.

**Crash fixed**
- Games could hang or crash when the NR feature was re-created (look preset change in OptiScaler, enabling NR, a history reset after loading). Cause: an out-of-bounds GPU read added with the halo fix.

**Faster**
- New c64 network kernel: about 20-25 % faster for that part (network about -4 %), output identical.

**Smaller changes**
- `split_fallback=1`: where the halo fix finds no matching old edit (e.g. a moving person), the area keeps the effect's light and tone instead of dropping to the raw frame.
- `int8_prepost=1`: post block MLP on int8 (slightly faster, image unchanged at 50 dB).

**Presets**
- New: `Hybrid_1440` (default), `Hybrid_1440_Fast` (nr_split=3), `Hybrid_1080`, `Hybrid_4K`, `Full_1440`, `Full_1080` (whole network at full size, most detail, slowest).
- The old presets are in `presets/classic` (720p/540p: much faster, but only a fraction of the effect).

**Measured** (Cyberpunk 2077, RX 6900 XT, 1440p, FSR Quality, Hybrid_1440): about 35 FPS with the full look, clean picture (minimal noise, minimal ghosting).

## v0.4.5 (2026-10-18)

Includes all v0.4.4 fixes (see below), plus a stronger look, debug tools and safer defaults. Same kernels and `nr_data`.

**Fixes**
- **History at 1280x720 now works.** The network pads 720p to 1344x768, and the history slot was sized for 1280x768, so history silently switched off at 720p (also in v0.4.4). With it on, 720p flicker excess drops from x1.17 to x1.00 in the test.
- **Recreate hang:** the sync shaders are compiled once per process (bytecode cache), not on every create (about 26 s each before). The exp/expB colour-transfer path is fixed too.

**Stronger look (closer to NVIDIA)**
- NVIDIA's style grade (Natural / Cinematic) is applied to the network output.
- `intensity_rule=linear` (default): intensity above 1 extends the edit linearly. `ratio` = the old rule.
- NVIDIA history rules: `history_gate=0`, `history_filter=catmull`, `history_pad=reflect`.
- New presets **Showcase_540p / Showcase_1080p** (style 2, intensity 2, local tone 2): about 5x the edit of Standard.

**Debug tools**
- `dump_frame=N` / `dump_frames=N` write frame inputs/outputs next to the log. `mv_scale` and a one-time `motion:` log line help check motion vectors.

**Defaults (package ini and all presets)**
- `nr_split=0`. **Known issue:** `nr_split=3` gives more fps, but the shown edit can visibly trail in game. This stays open until a two-frame capture explains it.
- `history_seed=0` (a per-frame seed caused white noise in game), `vit_every=1` (middle-stage reuse off).
- Package ini: `model_res=1280x720`, `edit_temporal=0.3`.

## v0.4.4 (2026-10-18)

Fixes from an internal code review, two merged features, a debug view and an investigation of the high model resolutions. Same kernels and `nr_data` as v0.4.
At the default settings the image is unchanged; in the deterministic tests the output is bit-identical to v0.4.3.

**Fixes**
- **Crash: a feature freed while the game used it.** If the game evaluated a feature at the moment another feature's load freed it as idle, v0.4.3 could use freed memory. Evaluate now checks again after it takes the lock and shows the native frame.
- **NR no longer ends for the session on a format or timing hiccup.** OptiScaler turns NR off for good when evaluate returns 0. Format, size and runtime conditions now always pass the live frame through and return success. When the colour and output formats differ (for example R11G11B10 to RGBA16F), the pass-through converts in a small shader instead of failing. It never sets a barrier on an output without UAV access, which could remove the device.
- **nr_split:** when the first part of a job never ran (a stall of more than 500 ms), its later parts are skipped. Before, they ran on stale data, and the D3D12 side waited for them.
- **`detail=` now works together with `edit_temporal`.** Before, it was silently ignored there.
- **`edit_temporal` now also works without `nr_split` / `history`** (motion vectors are bound for it on their own). If the game has no usable motion vectors, or the path cannot use it (game resolution = model resolution with `nr_split=0`), the log says so.
- **Frame-generation reuse frames** (`fg_mode=1`) now honour `compose=` / `detail=`.
- **R9G9B9E5 output (RE Engine):** a staging buffer is only freed once no frame in flight can still use it. Before, it was freed immediately when the size changed.
- **A new D3D12 device** (the game recreated it) gets its own HDR latch and readback buffers. Before, it reused the old device's buffers.
- **Async fallback:** intensity above 1 is capped at 1 there, as in the main path. Before, the edit was extrapolated linearly.
- **Self-check:** the startup check now compares the network output with that job's own input. v0.4.3 read the input buffer after the next frame could already have overwritten it, so during camera motion the logged `mean |out-in|` was too high (for example 0.13 in Fallout 4 at 1440p).

**New**
- **`debug_view`** (`nr_port.ini`): 1 shows the raw network output (upsampled), 2 shows the NR edit x4 + 0.5 (grey = no change). For bug reports.
- **The host's NR preset picks the model** (merged from branch `nr-preset`). With `model=auto`, OptiScaler's Preset 1 runs the old/large model and every other preset runs the new small one, as NVIDIA's DLL does. `model=small|large` still forces a model.
- **Colour transfer** (merged from branch `hdr-transfer`; RenoDX's "Control-compatible color transfer", MIT): `transfer_strength` (0 = off = as before), `color_strength`, and `paper_white` for HDR games.
- **AMDNR forwarder path:** the DLL now also exports the older `dlssnr_call_evaluate` (one shared depth/motion size) and `dlssnr_query_scaling_ratio` (always 1.0, because NR does not upscale), so hosts built on janblade's older forwarder can load it.

**High model resolutions (1080p / 1440p): what the artefacts are**
- Our network output matches the PyTorch reference implementation (dlssnr-pytorch-flex) at 540p, 1080p and 1440p, both at the default sliders and at style 2 / tone 2 / structure 2 / intensity 2. That is 37-42 dB, edit correlation 0.96-0.99, and the same edit strength. History matches too.
- At those sliders the reference itself changes the image about 3.5x as much as at the defaults (mean edit 0.055-0.06 vs 0.017), and flicker rises 4-5x. The strong, unstable look is NVIDIA's network at those settings, not the port. The edit strength is about the same at every model resolution, so 1080p / 1440p mostly add fine detail, not a stronger effect.
- The port's own path adds no resolution-dependent error. At model = game resolution it is bit-identical to the reference network per frame. With `history=1` and these strong sliders the history feeds the strong edit back and it builds up. That is the network's own recurrence, and the reference behaves the same. With strong sliders keep the preset settings (`edit_temporal=0.7`, `nr_split=3`): in the test, flicker at 1080p / 1440p drops from 7.7 / 11.5 % to 1.7 / 2.6 % of pixels.

## v0.4 (public release, 2026-10-17)

This public release contains everything from the internal builds v0.3.1, v0.4, v0.4.1, v0.4.2 and v0.4.3 listed below.
Known issues: see README ("Known issues (v0.4)").

## v0.4.3 (internal)

- **Looks to choose from.** `presets\` now holds full `nr_port.ini` files, each with a short note on look, cost and when to use it: Standard (the default, = `nr_port.ini`), Strong, Cinematic, HighQuality_720p, HiRes_1080p, Native_1440p, Performance, Community. To switch, copy one over `nr_port.ini`. See "Choosing a look" in the README.
- **Intensity above 1 keeps the colours.** NVIDIA's rule: the brightness ratio is raised to the intensity (at most 2x either way) and the hue stays. Before, the edit was scaled per channel, which shifted colours (about 2x the chroma shift at intensity 2). Intensity up to 1 gives the v0.4.2 image.
- **Strong** (`local_tone=2`, `intensity=1.3`): about +45 % effect, shadows +41 %, faces +35-85 %, no extra GPU time. **Cinematic** = Strong + NVIDIA's Cinematic style: the strongest, graded look.
- **New `nr_port.ini` keys** `compose=add|nv|ratio` (default `add` = as before; `nv` = NVIDIA's own blend, for users copying NVIDIA multipass setups; `ratio` = OkLab residual after janblade's Transfer 2) and `detail=` (live-frame detail above the model resolution, 1 = as is). The **Community** preset uses `compose=nv` and `pass2_gain=1`.
- **Much less flicker: `edit_temporal=0.7`** (new default in `nr_port.ini` and every preset). The shown NR edit is blended with the previous one, moved by the game's motion vectors, depth-gated and clamped to the local range of the current edit (two small model-size passes, no measurable cost). Flicker harness (2560x1440 frame, settled frames, 960x540 / 1920x1080 model): Standard 0.83 -> 0.075 % / 0.68 -> 0.090 %, Strong 1.48 -> 0.20 % / 1.05 -> 0.16 %, Cinematic 4.26 -> 1.46 % / 2.71 -> 0.83 % visible flicker. `edit_temporal=0` gives the v0.4.2 image. Also new: `edit_limit` (cap the edit against local contrast, 0 = off) and `edit_warp=bicubic`.
- Same kernels and `nr_data` as v0.4.2.

## v0.4.2 (internal)

- **Faster, same image.** New c32 swin kernel (one window per 64-thread workgroup, weights from a load-time f16 shadow, packed q8/e4m3; about 2.1x faster) and new ViT GEMMs (8x4 register tile, 128 threads, LDS prefetch). Bit-exact with v0.4.1 (both models, 960x540 / 1280x720 / 832x468 / 640x360). The c32 kernel is also deterministic run to run now. Network at 960x540 / 1280x720 (RX 6900 XT, median, new model): 11.02 -> 9.67 ms / 17.55 -> 15.03 ms; old model 20.26 -> 18.04 ms / 32.65 -> 28.36 ms.
- **`nr_split=N` (2, 3 or 4).** The network runs in N balanced parts over N frames (split points measured at load); the shown edit is the last finished one, warped by the game's motion vectors. New default `nr_split=3` (`nr_split=1` still means 2 parts, `0` = off); the Performance preset uses 4. Synthetic pacing test (2560x1440 frame, 960x540, new model, median fps): CPU 8 / GPU 12 ms per frame: 53 / 58 / 60 fps at N = 2 / 3 / 4; CPU 6 / GPU 6 ms: 79 / 88 / 110 fps.
- **Model resolutions 1920x1080 and 2560x1440** (new model; old model 1920x1080) for image-quality comparison: presets `HiRes_1080p.ini` / `Native_1440p.ini`. Network (RX 6900 XT, vit_every=1): 9.7 / 14.9 / 29.5 / 51.2 ms at 540p / 720p / 1080p / 1440p (old model 1080p 54.7 ms); VRAM about 0.86 GB at 1080p, 1.42 GB at 1440p. `model_res=auto` never picks them (only an explicit `model_res`). 1080p data uses NVIDIA's field 1088 rows (was 1152: +0.4 dB vs the reference).
- **Fix: block artifacts when the game resolution equals the model resolution** (e.g. 1280x720 game at 1280x720, or the 1440p preset): the network input is now clamped to [0,1] there too (as for every other size), and highlights above 1 keep their brightness (frame + NR edit).
- The data dirs are new (c32 workgroup size changed): use the `nr_data` from this package, not an older one.

## v0.4.1 (internal)

- **Faster, same image.** The pre and post blocks now read the game's colour and write the result themselves (k_import, the mirror padding and k_export are fused in), and the post block passes its arguments by value. Bit-exact with v0.4. Network at 960x540 / 1280x720 (RX 6900 XT, median): new model 11.33 -> 10.97 ms / 18.03 -> 17.47 ms, old model 20.68 -> 20.31 ms / 33.25 -> 32.70 ms.
- **ViT reuse default per model.** `vit_every=0` (new default in `nr_port.ini` and the Quality preset) uses the model's own default: every 3rd job for the new model, every 2nd for the old one (it flickers more at 3). Any other value still overrides it.
- New opt-in `int8_prepost=1`: post-block MLP on int8 dot products (about -0.1 ms, tiny image change). Default 0.

## v0.4.0 (internal)

- **NVIDIA's new model.** nvngx_dlssnr.dll 310.9.16 carries a new, ~2x lighter model (`WEIGHTS_CC_SMALL_FP8`). It is picked automatically; `model=large` in `nr_port.ini` forces the old one. 310.8 keeps working with the old model. Network at 960x540 (RX 6900 XT): **11.1 ms** (new model) vs 24.3 ms in v0.3.1; with the default settings about 7 ms.
- **Much closer to NVIDIA's image** (+6-8 dB vs the reference implementation): window-shift axis, decoder phase, ViT extent, NVIDIA's geometry (`Geometry::fromValid`), mirror padding and pooling layout fixed; NaN outputs become +0. The window-shift fix also makes every swin step cover the full plane, which removes the old rare (about 1 in 420 runs) nondeterminism of the c32/swin kernels (stale rows 508-511).
- **NVIDIA's default look**: intensity 1, style 0, local tone 1, local structure 1, skin -1, auto mask on.
- **The OptiScaler NR sliders work** (intensity, style, local tone, local structure, skin, auto mask), live. `nr_port.ini` keys of the same name override them.
- **Faster kernels**: int8 swin MLP/QKV (`int8_swin=1`), faster pre/post blocks and softmax, post weights built at load (so the post block also serves the new model), ViT QKV register tiling.
- **Presets** in `presets\`: Quality (960x540, ViT every 3rd job, new default), Balanced (832x468), Performance (640x360).
- `nr_data` now ships per model: `nr_data\<model id>\<WxH>\` for 960x540, 1280x720, 832x468 and 640x360.

## v0.3.1 (internal)

- **Uses the correct GPU.** NR runs on the GPU the game renders on (matched by adapter LUID), never on an iGPU.
- **VRAM leak fix** (MSFS 2024): features the game abandons are freed, and recreating NR no longer grows VRAM use.
- **HDR auto-exposure** (RE Requiem): scene-linear HDR input is exposed automatically before the network.
- **HDR with 2 passes**: the second pass follows the first pass's HDR decision, and its brightening is capped.
- **Unknown or new `nvngx_dlssnr.dll`**: NR turns itself off cleanly and the game keeps its normal image; `dlssnr_nr.log` says why.
- **All kernels are our own**, with no NVIDIA data inside; the weights are read from your `nvngx_dlssnr.dll` at load.
- **Faster**: about −0.6 ms per network run.
- **Less VRAM**: our own frame plan needs about 380 MB instead of 620 MB at 960x540 (490 MB instead of 860 MB at 1280x720).
- New `nr_port.ini` keys (defaults): `hdr_key=0.18`, `hdr_adapt=0.05`, `hdr_exposure_bias=1.0`, `hdr_exposure=0`, `allow_any_gpu=0`, `evict_idle_ms=2000`.

## v0.3

- **Up to 60 FPS without frame generation** on an RX 6900 XT (Fallout 4, 1440p, 960x540 model): about 40-45 FPS in heavy scenes, up to 60 in lighter ones.
- **Even frame pacing.** `nr_split=1` splits the network over two frames, so the frame time stays even and no longer alternates long/short.
- **Faster.** The c32 kernels now run natively.
- **Less flicker.** The temporal history follows the game's motion vectors.
- **Fast games no longer stutter.** v0.2.3 dropped to a native frame whenever the GPU was still busy (Assetto Corsa). Now it waits for the GPU instead (`busy_mode=wait`, `busy_wait_ms=100`). In the pacing test this gives 0 native frames.
- **HDR fix.** Characters and highlights no longer turn white in HDR games (RE Engine). `hdr_mode=auto` detects HDR input and converts it for the network.
- **Clean 2 passes.** With OptiScaler's "2 passes" the effect is no longer applied twice. The second pass is damped (`pass2_gain=0.5`).
- **Safe setup and uninstall.** `setup_windows.bat` backs up every file it replaces, and `Uninstall_DLSS5.bat` restores them. The setup text is now AMD-specific.
- **Wrong-folder warning.** Setup warns when the folder has no 64-bit game exe, for example a 32-bit launcher folder.
- **Hardware log and self-check.** `dlssnr_nr.log` records the GPU, driver and VRAM, warns when VRAM is low, and runs a self-check at start. If the self-check fails, NR switches off cleanly instead of crashing.
- **ShortFuse's nvngx_dlssnr.dll works.** The weights are found by name, so any build that carries the same weights works.
- New `nr_port.ini` defaults: `nr_split=1`, `busy_mode=wait`, `hdr_mode=auto`, `pass2_gain=0.5`.
- `nr_data` ships 960x540 (recommended) and 1280x720.

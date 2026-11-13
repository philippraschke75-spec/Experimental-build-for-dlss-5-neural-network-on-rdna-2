# Rdna2NrRuntime for OpenNR

The RDNA2 (RX 6000) port as an optional runtime behind the lmxxf runtime C ABI v1 that the OpenNR host loads.

- Sole export `LmxxfNrGetApi(1, &api)` with the 136-byte table (EnqueueHip + Poll).
- `LmxxfNrFrameInfo` per OpenNR's header (struct_size 64..152), `LmxxfNrJob` 24 or 32 bytes.
- Uses STRENGTH (capped at 1), DEBUG_VIEW, CODEC_PASSTHROUGH, TEMPORAL (motion + reset), passes, model_scale;
  other runtimes' flags are ignored and logged once.
- Sync: shared D3D12 buffers imported into HIP, bounded waits on both sides, fallback to the live frame.
- Needs `nr_runtime.dll` (built from `src/runtime`), the `nr_data` folder from the release, and the user's own
  `nvngx_dlssnr.dll` for the weights. Settings: `Rdna2NrRuntime.ini`.

Tested 2026-11-13 on an RX 6900 XT through a harness that plays the OpenNR host: 1707x960 12.0 ms, 2560x1440 25.2 ms.

Build (MSVC x64): `cl /LD /O2 /EHsc /std:c++17 rdna2_nr_runtime.cpp /Fe:Rdna2NrRuntime.dll /link d3d12.lib dxgi.lib d3dcompiler.lib`

// Interface of the drop-in nvngx.dll_dlssnr.dll ("vendor-neutral port" flavour), as OptiScaler-NR loads it.
// Recovered from the public OptiScaler-NR source (janblade/OptiScaler-DLSSNR-PreSR-Multipass):
//   OptiScaler/dlssnr/forwarder/dlssnr_forwarder.cpp  (export definitions, D3D12 path)
//   OptiScaler/shaders/dlssnr/DlssNr_Dx12.cpp         (loader EnsureForwarder(), PFN_Nr* typedefs, call sites)
// See INTERFACE.md for semantics. All exports are extern "C", x64 (one calling convention, __cdecl == default).
#pragma once
#include <d3d12.h>

// Host resolves these by GetProcAddress on the module named "nvngx.dll_dlssnr.dll" (file name must be exactly that).
using PFN_NrCreate = void*(__cdecl*)(const wchar_t* snippetPath,   // full path of the user-supplied nvngx_dlssnr.dll (weights/runtime)
                                     const wchar_t* dataPath,      // State::NVNGX_ApplicationDataPath
                                     ID3D12Device* device,
                                     ID3D12GraphicsCommandList* cmd, // recording list; init work may be recorded into it
                                     void* capabilityParams,       // NGX parameter block; nullptr for the port (host skips the NGX core)
                                     unsigned int width, unsigned int height, // model working size
                                     int preset, float intensity, int style,
                                     float localStructure, float localTone, float skinStructure,
                                     int useAutoMask, int uiCorrection);

using PFN_NrEvaluate = int(__cdecl*)(ID3D12GraphicsCommandList* cmd, void* feature,
                                     void* capabilityParams,       // nullptr for the port
                                     ID3D12Resource* color, ID3D12Resource* depth,
                                     ID3D12Resource* motion, ID3D12Resource* output,
                                     unsigned int width, unsigned int height,             // model working size
                                     unsigned int guideWidth, unsigned int guideHeight,   // depth valid size
                                     unsigned int motionWidth, unsigned int motionHeight, // motion valid size
                                     unsigned int depthBaseX, unsigned int depthBaseY,
                                     unsigned int motionBaseX, unsigned int motionBaseY,
                                     int depthInverted, int reset,
                                     float intensity, int style, float localStructure, float localTone,
                                     float skinStructure, int useAutoMask,
                                     float mvScaleX, float mvScaleY);   // returns 1 = NVSDK_NGX_Result_Success

using PFN_NrRelease = void(__cdecl*)(void* feature);
using PFN_NrSetExtras = void(__cdecl*)(void* capabilityParams, float globalTone, ID3D12Resource* ui,
                                       ID3D12Resource* uiAlpha, ID3D12Resource* backbuffer,
                                       unsigned int uiWidth, unsigned int uiHeight,
                                       unsigned int bbWidth, unsigned int bbHeight);
using PFN_NrSetFloatSlot = void(__cdecl*)(int slot);
using PFN_NrProbeFloat = void(__cdecl*)(void* params, const char* name, float value, int slot);
using PFN_NrError = const char*(__cdecl*)();

// Data exports (host reads them through GetProcAddress as int*):
//   int dlssnr_call_last_init, dlssnr_call_last_create, dlssnr_last_ratio_result, dlssnr_last_ratio_stage;
// Presence-only export (its type is never called by the host; the real port exports the name):
//   dlssnr_backend_id

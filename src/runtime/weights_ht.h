// WEIGHTS_HT resource of NVIDIA's nvngx_dlssnr.dll (layout: rdna2/emu/weights_ht.py). Shared by nr_runtime (weights.map
// patching) and the DLL (model detection). Windows only, header only.
#pragma once
#include <windows.h>
#include <cstdint>
#include <cstring>
#include <string>

// Loads `path` as a data file and calls cb(name, data, bytes) per record. Returns the record count (0 = no DLL / no resource).
// The module stays loaded while cb runs and is freed afterwards.
template <class F> static int nr_weights_ht(const std::string& path, const char* res, F cb) {   // res: WEIGHTS_HT (310.8 model) or WEIGHTS_CC_SMALL_FP8 (310.9 small), same framing
  HMODULE dll = LoadLibraryExA(path.c_str(), nullptr, LOAD_LIBRARY_AS_DATAFILE | LOAD_LIBRARY_AS_IMAGE_RESOURCE);
  if (!dll) return 0;
  HRSRC r = FindResourceA(dll, res, MAKEINTRESOURCEA(10));
  const uint8_t* b = r ? (const uint8_t*)LockResource(LoadResource(dll, r)) : nullptr; uint64_t sz = r ? SizeofResource(dll, r) : 0, o = 8; int n = 0;
  while (b && o + 8 <= sz) {   // record: u64 name_len, name, u64, u64, u64 D, u32 1, data[D], 20-byte trailer
    uint64_t L, D; memcpy(&L, b + o, 8); if (o + 8 + L + 28 > sz) break;
    std::string nm((const char*)b + o + 8, (size_t)L); memcpy(&D, b + o + 8 + L + 16, 8);
    o += 8 + L + 28; if (o + D + 20 > sz) break; cb(nm, b + o, D); o += D + 20; n++;
  }
  FreeLibrary(dll); return n;
}
// Model id = FNV-1a 64 over every record's name, a 0 byte and its u64 size (the architecture, not the weight values).
// Same as model_id() in handover/game-ready-sync/runtime/model_id.py. 0 = no DLL / no WEIGHTS_HT.
static uint64_t nr_model_id(const std::string& dll_path, const char* res = "WEIGHTS_HT") {
  uint64_t h = 0xcbf29ce484222325ull;
  auto mix = [&](const void* p, size_t n) { for (size_t i = 0; i < n; i++) h = (h ^ ((const uint8_t*)p)[i]) * 0x100000001b3ull; };
  int n = nr_weights_ht(dll_path, res, [&](const std::string& nm, const uint8_t*, uint64_t d) { mix(nm.c_str(), nm.size() + 1); mix(&d, 8); });
  return n ? h : 0;
}

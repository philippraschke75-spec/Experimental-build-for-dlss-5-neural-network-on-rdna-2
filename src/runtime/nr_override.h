// Manifest step overrides (see OVERRIDES.md). Shared by net_run.cpp and nr_runtime.cpp.
// No override file -> empty map -> callers behave exactly as before.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct NrOverride {
  std::string mod, sym;       // native module path / symbol
  std::string g[4];           // gx gy gz threads; "=" keeps the manifest value
  size_t len = 0;             // new kernarg size (zero-filled, then recipe)
  struct Op { size_t dst, src, n; bool lit; uint8_t v[8]; };
  std::vector<Op> ops;
};

// overrides file path: $NR_OVERRIDES, else overrides.txt next to the manifest. Returns false (with msg) on a malformed file.
static bool nr_load_overrides(const std::string& manifest, std::map<std::string, NrOverride>& out, std::string& msg) {
  std::string p;
  if (const char* e = getenv("NR_OVERRIDES")) p = e;
  else { size_t s = manifest.find_last_of("/\\"); p = (s == std::string::npos ? std::string() : manifest.substr(0, s + 1)) + "overrides.txt"; }
  FILE* f = fopen(p.c_str(), "r"); if (!f) { if (getenv("NR_OVERRIDES")) { msg = "cannot open " + p; return false; } return true; }
  char line[4096];
  while (fgets(line, sizeof line, f)) {
    std::string s(line); while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    if (s.empty() || s[0] == '#') continue;
    std::vector<std::string> t; size_t pos = 0;
    for (;;) { size_t q = s.find('|', pos); if (q == std::string::npos) { t.push_back(s.substr(pos)); break; } t.push_back(s.substr(pos, q - pos)); pos = q + 1; }
    if (t.size() != 9) { fclose(f); msg = "bad override line (need 9 fields): " + s; return false; }
    NrOverride o; o.mod = t[1]; o.sym = t[2]; for (int i = 0; i < 4; i++) o.g[i] = t[3 + i];
    o.len = strtoull(t[7].c_str(), nullptr, 0);
    const char* r = t[8].c_str();   // recipe: space separated "dst<-src:n" or "dst=u32:v" / "dst=u64:v" / "dst=f32:v"
    while (*r) {
      while (*r == ' ') r++; if (!*r) break;
      char* e; NrOverride::Op op{}; op.dst = strtoull(r, &e, 0);
      if (e[0] == '<' && e[1] == '-') { op.src = strtoull(e + 2, &e, 0); if (*e != ':') break; op.n = strtoull(e + 1, &e, 0); }
      else if (e[0] == '=' && e[4] == ':' ) {
        op.lit = true; std::string ty(e + 1, 3); const char* v = e + 5;
        if (ty == "u32") { uint32_t x = (uint32_t)strtoul(v, &e, 0); memcpy(op.v, &x, 4); op.n = 4; }
        else if (ty == "u64") { uint64_t x = strtoull(v, &e, 0); memcpy(op.v, &x, 8); op.n = 8; }
        else if (ty == "f32") { float x = strtof(v, &e); memcpy(op.v, &x, 4); op.n = 4; }
        else break;
      } else break;
      if (*e && *e != ' ') break;
      if (op.dst + op.n > o.len) { fclose(f); msg = "override recipe writes past ka_len: " + s; return false; }
      o.ops.push_back(op); r = e;
    }
    if (*r) { fclose(f); msg = std::string("bad recipe item at '") + r + "': " + s; return false; }
    out[t[0]] = o;
  }
  fclose(f); return true;
}

// Build the native kernarg from the (already rebased) original one. Copies past the source end fail.
static bool nr_apply_override(const NrOverride& o, const std::vector<uint8_t>& src, std::vector<uint8_t>& ka) {
  ka.assign(o.len, 0);
  for (auto& op : o.ops) {
    if (op.lit) memcpy(ka.data() + op.dst, op.v, op.n);
    else { if (op.src + op.n > src.size()) return false; memcpy(ka.data() + op.dst, src.data() + op.src, op.n); }
  }
  return true;
}
static unsigned nr_grid(const std::string& v, unsigned orig) { return v == "=" ? orig : (unsigned)strtoul(v.c_str(), nullptr, 0); }

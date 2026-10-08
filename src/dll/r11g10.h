// R11G11B10_FLOAT <-> RGBA16F (half4). Unsigned small floats: 5-bit exponent, bias 15, same as half, so unpack is a bit shuffle.
#pragma once
#include <cstdint>
static inline uint32_t half_to_small(uint16_t h, int mant_bits) {   // half -> unsigned float with mant_bits (6 = R/G, 5 = B); negative/-0 -> 0, inf/NaN/overflow -> max finite
    if (h & 0x8000) return 0;
    int sh = 10 - mant_bits; uint32_t v = h & 0x7FFF;
    v = (v + ((1u << (sh - 1)) - 1) + ((v >> sh) & 1)) >> sh;         // round to nearest even (carry may bump the exponent)
    uint32_t mx = (30u << mant_bits) | ((1u << mant_bits) - 1);
    return v > mx ? mx : v;
}
static inline void pack_r11g11b10(const uint16_t* px /*4 halves*/, uint32_t* out) {
    *out = half_to_small(px[0], 6) | (half_to_small(px[1], 6) << 11) | (half_to_small(px[2], 5) << 22);
}
static inline void unpack_r11g11b10(uint32_t v, uint16_t* px /*4 halves*/) {
    px[0] = (uint16_t)((v & 0x7FF) << 4); px[1] = (uint16_t)(((v >> 11) & 0x7FF) << 4); px[2] = (uint16_t)(((v >> 22) & 0x3FF) << 5); px[3] = 0x3C00;
}

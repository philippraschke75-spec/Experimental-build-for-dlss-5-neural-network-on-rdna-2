#!/bin/sh
# build.sh [outdir]: the native kernel modules of the v0.5.2 plan (gfx1030; every RDNA2 chip loads them, see nr_runtime).
# v0.5.2: fast FP8 emulation (-DQ8FAST -DQ8BYTE), FAST_ATTN, and per module the faster of CU / WGP mode and
# -ffp-contract=off / -ffast-math (measured at 1440p on an RX 6900 XT, ROCm 6.4).
# pre needs a vendor-derived table (pre_tables.h, never committed) and is built only if it is present.
R="/c/Program Files/AMD/ROCm/6.4"; cd "$(dirname "$0")"; O="${1:-.}"
B="-x hip --offload-arch=gfx1030 -O3 --cuda-device-only -Wno-ignored-attributes -std=c++17 -cuid=dlssnr -I. -DQ8FAST -DQ8BYTE -DFAST_ATTN"
CU="-mcumode -ffp-contract=off"; WGP="-ffp-contract=off"; CUF="-mcumode -ffast-math"; WGPF="-ffast-math"
c() { n=$1; s=$2; shift 2; "$R/bin/clang++.exe" -I"$R/include" $B "$@" -c -o "$O/native_$n.co" "$s.hip" || exit 1; }
c c512 c512 $CU
c decup decup $CU
c mirror mirror $CU
c resample resample $CU
c vit vit_swz $WGP
c swinvar swin_var $WGP
c swin32 swin_c32 $WGP
c swin64n swin_c64n $CUF -DWPE=2
c io io_head $CUF
c post_native post_native $CU -DPREW=0 "-DPOST_ATTR=__attribute__((amdgpu_waves_per_eu(6)))"
[ -f pre_tables.h ] && c pre_native pre_native $WGPF "-DPRE_ATTR=__attribute__((amdgpu_waves_per_eu(7)))"
exit 0

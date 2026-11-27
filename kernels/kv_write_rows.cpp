// Prefill: write T token rows of K and V into a (non-paged) cache [1, NKV, S, HD] (bfp8_b or bf16), one
// read-modify-write per destination cache tile. The sources are head-major [1, NKV, T, HD] bf16 tiles (32 tokens per
// tile row), POS (int32 row-major [T]) the cache row of each token (padding rows: a scratch row).
// Units: (kv, head h, column tile c, group g), a group = a run of consecutive tokens whose rows share a cache tile;
// core i takes the units u = i, i + ncores, ... (same order on every core). A unit reads the cache tile, the source
// tile(s) of its tokens, rewrites the tokens' rows and writes the tile back. Distinct groups have distinct cache tiles.
// bfp8_b tile (1088 B): 64 exponent bytes (face f = (row/16)*2 + col/16, index f*16 + row%16), then 4 faces x 16 rows
// x 16 mantissa bytes (as kv_write.cpp: round to nearest even).
// CT: 0 T, 1 NKV, 2 HDT, 3 S (cache rows), 4 pos bytes, 5 BFP8, accessors Ks, Vs, Kc, Vc, POS
// RT: 0..4 addrs Ks, Vs, Kc, Vc, POS; 5 core index, 6 ncores
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t T = get_compile_time_arg_val(0), NKV = get_compile_time_arg_val(1), HDT = get_compile_time_arg_val(2);
constexpr uint32_t S = get_compile_time_arg_val(3), POSB = get_compile_time_arg_val(4), BFP8 = get_compile_time_arg_val(5);

FORCE_INLINE uint32_t elem_off(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

FORCE_INLINE void to_bfp8(const uint16_t* v, uint8_t* exp_out, uint8_t* man_out) {
    uint32_t se = 0;
    for (uint32_t i = 0; i < 16; ++i) {
        const uint32_t e = (v[i] >> 7) & 0xff;
        se = e > se ? e : se;
    }
    *exp_out = (uint8_t)se;
    for (uint32_t i = 0; i < 16; ++i) {
        const uint32_t b = v[i];
        const uint32_t e = (b >> 7) & 0xff;
        uint32_t out = 0;
        if (e != 0) {
            uint32_t m = (1u << 23) | ((b & 0x7f) << 16);
            const uint32_t d = se - e;
            m = d > 31 ? 0 : (m >> d);
            const uint32_t rnd = m & 0x1ffff;
            m >>= 17;
            if (rnd > 0x10000 || (rnd == 0x10000 && (m & 1))) m += 1;
            if (m > 127) m = 127;
            out = m ? (((b >> 15) & 1) << 7) | m : 0;
        }
        man_out[i] = (uint8_t)out;
    }
}

void kernel_main() {
    constexpr auto aKs = TensorAccessorArgs<6>();
    constexpr auto aVs = TensorAccessorArgs<aKs.next_compile_time_args_offset()>();
    constexpr auto aKc = TensorAccessorArgs<aVs.next_compile_time_args_offset()>();
    constexpr auto aVc = TensorAccessorArgs<aKc.next_compile_time_args_offset()>();
    constexpr auto aP = TensorAccessorArgs<aVc.next_compile_time_args_offset()>();
    const auto sKs = TensorAccessor(aKs, get_arg_val<uint32_t>(0));
    const auto sVs = TensorAccessor(aVs, get_arg_val<uint32_t>(1));
    const auto sKc = TensorAccessor(aKc, get_arg_val<uint32_t>(2));
    const auto sVc = TensorAccessor(aVc, get_arg_val<uint32_t>(3));
    const auto sP = TensorAccessor(aP, get_arg_val<uint32_t>(4));
    const uint32_t ci = get_arg_val<uint32_t>(5), nc = get_arg_val<uint32_t>(6), dbg = get_arg_val<uint32_t>(7);
    constexpr uint32_t TB = 2048, CTB = BFP8 ? 1088 : 2048, MT = T / 32;
    CircularBuffer cpos(0), csrc(1), cdst(2), cgrp(3);
    cpos.reserve_back(1);
    csrc.reserve_back(1);  // one page of two tiles
    cdst.reserve_back(1);
    cgrp.reserve_back(1);
    const uint32_t sb = csrc.get_write_ptr(), db = cdst.get_write_ptr();
    noc_async_read(sP.get_noc_addr(0), cpos.get_write_ptr(), POSB);
    noc_async_read_barrier();
    const int32_t* pos = reinterpret_cast<const int32_t*>(cpos.get_write_ptr());
    if (dbg & 1) return;
    // groups: [start token, cache row tile] for runs of tokens on one cache tile
    uint16_t* g0 = reinterpret_cast<uint16_t*>(cgrp.get_write_ptr());
    uint16_t* gt = g0 + T + 1;
    uint32_t G = 0;
    for (uint32_t t = 0; t < T; ++t) {
        const uint32_t rt = (uint32_t)pos[t] / 32;
        if (G == 0 || gt[G - 1] != rt) {
            g0[G] = t;
            gt[G++] = rt;
        }
    }
    g0[G] = T;
    const uint32_t NU = (dbg & 2) ? 0 : 2 * NKV * HDT * G;
    for (uint32_t u = ci; u < NU; u += nc) {
        const uint32_t g = u % G, c = (u / G) % HDT, h = (u / G / HDT) % NKV, kv = u / G / HDT / NKV;
        const uint32_t ta = g0[g], tb = g0[g + 1];
        const uint32_t cache_tile = (h * (S / 32) + gt[g]) * HDT + c;
        const auto& sc = kv ? sVc : sKc;
        const auto& ss = kv ? sVs : sKs;
        noc_async_read(sc.get_noc_addr(cache_tile), db, CTB);
        noc_async_read_barrier();
        const uint16_t* src = reinterpret_cast<const uint16_t*>(sb);
        uint32_t ms = 0xFFFFFFFF;  // source row tile in sb (a group can span many: the padding rows share one row)
        for (uint32_t t = ta; t < tb && !(dbg & 4); ++t) {
            if (t / 32 != ms) {
                ms = t / 32;
                noc_async_read(ss.get_noc_addr((h * MT + ms) * HDT + c), sb, TB);
                noc_async_read_barrier();
            }
            const uint16_t* st = src;
            const uint32_t r = t & 31, rr = (uint32_t)pos[t] % 32;
            if constexpr (BFP8) {
                uint8_t* d8 = reinterpret_cast<uint8_t*>(db);
                for (uint32_t half = 0; half < 2; ++half) {
                    uint16_t v[16];
                    for (uint32_t i = 0; i < 16; ++i) v[i] = st[elem_off(r, half * 16 + i)];
                    const uint32_t f = (rr >> 4) * 2 + half;
                    to_bfp8(v, d8 + f * 16 + (rr & 15), d8 + 64 + f * 256 + (rr & 15) * 16);
                }
            } else {
                uint16_t* d = reinterpret_cast<uint16_t*>(db);
                for (uint32_t col = 0; col < 32; ++col) d[elem_off(rr, col)] = st[elem_off(r, col)];
            }
        }
        noc_async_write(db, sc.get_noc_addr(cache_tile), CTB);
        noc_async_write_barrier();
    }
}

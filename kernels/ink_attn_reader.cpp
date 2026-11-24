// Inkling decode attention (one token) — reader. Split-K flash-decoding like gqa_attn_reader.cpp, over a non-paged
// cache [1, NKV, BSL, HD] (bfp8_b / bf16), with the mask of every chunk built here:
//   ring (sliding layers): slot j < R holds the newest position with that slot; distance d = (c - j) mod R
//     (c = P0 mod R); valid if d <= min(P0, W - 1); value = REL[h, d]
//   full layers: key j <= P0; d = P0 - j; value = REL[h, d] if d < W else 0
//   slot SINK (K = V = 0, never written): value = SINKV[h] (the per-token sink logit); everything else masked.
// Values are in units of 1 / scale (the compute multiplies mask + QK^T by the softmax scale).
// Chunks (C = CT * 32 keys): the n_norm chunks covering [0, ring ? min(P0 + 1, R) : P0 + 1), then the chunk of
// SINK if it is not among them; split s takes a contiguous range of that list (as gqa_attn).
// Q rows: row r < GH = q head g * GH + r (one token, QMODE 0 gather); rows >= GH fully masked (ignored downstream).
//   Q    : [1, 1, NH, HD] bf16 TILE (not pre-scaled)
//   REL  : [1, 1, NH, WR] bf16 TILE, the relative bias by distance (WR = R for the ring, W for full layers)
//   SINKV: [1, 1, 1, NH] bf16 TILE (row 0, column h)
//   POS  : int32 [8], POS[0] = P0
// CT: 0 RT(=1), 1 CT, 2 S, 3 NKV, 4 GH, 5 HDT, 6 KVTB, 7 BSL, 8 RING, 9 R, 10 W, 11 SINK, 12 WRT (REL column tiles),
//     then accessors Q, Kc, Vc, POS, SC, CF, REL, SINKV
// RT: 0..7 addrs, 8 g, 9 s
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t c_Q = 0, c_K = 1, c_V = 2, c_MASK = 3, c_SC = 4, c_INFO = 5, c_CONST = 19, c_SCR = 20, c_REL = 21;

FORCE_INLINE uint32_t elem_off(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

void kernel_main() {
    constexpr uint32_t RT = get_compile_time_arg_val(0);
    constexpr uint32_t CT = get_compile_time_arg_val(1);
    constexpr uint32_t S = get_compile_time_arg_val(2);
    constexpr uint32_t NKV = get_compile_time_arg_val(3);
    constexpr uint32_t GH = get_compile_time_arg_val(4);
    constexpr uint32_t HDT = get_compile_time_arg_val(5);
    constexpr uint32_t KVTB = get_compile_time_arg_val(6);
    constexpr uint32_t BSL = get_compile_time_arg_val(7);
    constexpr uint32_t RING = get_compile_time_arg_val(8);
    constexpr uint32_t R = get_compile_time_arg_val(9);
    constexpr uint32_t W = get_compile_time_arg_val(10);
    constexpr uint32_t SINK = get_compile_time_arg_val(11);
    constexpr uint32_t WRT = get_compile_time_arg_val(12);
    constexpr auto aQ = TensorAccessorArgs<13>();
    constexpr auto aK = TensorAccessorArgs<aQ.next_compile_time_args_offset()>();
    constexpr auto aV = TensorAccessorArgs<aK.next_compile_time_args_offset()>();
    constexpr auto aP = TensorAccessorArgs<aV.next_compile_time_args_offset()>();
    constexpr auto aSC = TensorAccessorArgs<aP.next_compile_time_args_offset()>();
    constexpr auto aCF = TensorAccessorArgs<aSC.next_compile_time_args_offset()>();
    constexpr auto aREL = TensorAccessorArgs<aCF.next_compile_time_args_offset()>();
    constexpr auto aSK = TensorAccessorArgs<aREL.next_compile_time_args_offset()>();
    const auto sQ = TensorAccessor(aQ, get_arg_val<uint32_t>(0));
    const auto sK = TensorAccessor(aK, get_arg_val<uint32_t>(1));
    const auto sV = TensorAccessor(aV, get_arg_val<uint32_t>(2));
    const auto sP = TensorAccessor(aP, get_arg_val<uint32_t>(3));
    const auto sSC = TensorAccessor(aSC, get_arg_val<uint32_t>(4));
    const auto sCF = TensorAccessor(aCF, get_arg_val<uint32_t>(5));
    const auto sREL = TensorAccessor(aREL, get_arg_val<uint32_t>(6));
    const auto sSK = TensorAccessor(aSK, get_arg_val<uint32_t>(7));
    const uint32_t g = get_arg_val<uint32_t>(8);
    const uint32_t s = get_arg_val<uint32_t>(9);

    constexpr uint32_t TB = 2048;
    constexpr uint32_t C = CT * 32;
    constexpr uint16_t NEG = 0xC6EA;  // bf16 -29952
    Noc noc;

    CircularBuffer scr(c_SCR);
    scr.reserve_back(HDT);
    const uint32_t sbase = scr.get_write_ptr();
    noc.async_read(sP, scr, 32, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    const uint32_t P0 = (uint32_t)(reinterpret_cast<volatile tt_l1_ptr int32_t*>(sbase)[0]);
    const uint32_t nn = RING ? (P0 + 1 < R ? P0 + 1 : R) : P0 + 1;
    const uint32_t n_norm = (nn + C - 1) / C;
    constexpr uint32_t SBASE = (SINK / C) * C;
    const uint32_t nch = n_norm + (SBASE >= n_norm * C ? 1 : 0);
    const uint32_t base_n = nch / S, rem = nch % S;
    const uint32_t c0 = s * base_n + (s < rem ? s : rem);
    const uint32_t c1 = c0 + base_n + (s < rem ? 1 : 0);
    const uint32_t cpos = P0 % R;
    const uint32_t dmax = P0 < W - 1 ? P0 : W - 1;

    CircularBuffer info(c_INFO);
    info.reserve_back(1);
    {
        volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(info.get_write_ptr());
        w[0] = c1 - c0;
        w[1] = c0;
        w[2] = P0;
    }
    info.push_back(1);

    CircularBuffer sc(c_SC), cst(c_CONST), cq(c_Q), crel(c_REL);
    sc.reserve_back(1);
    noc.async_read(sSC, sc, TB, {.page_id = 0}, {.offset_bytes = 0});
    cst.reserve_back(2);
    noc.async_read(sCF, cst, 4096, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read(sCF, cst, 4096, {.page_id = 1}, {.offset_bytes = 4096});
    const bool active = c1 > c0;
    // the relative bias (+ the sink tile at the end) of this core's heads, when any of its chunks needs it
    bool need_rel = false;
    for (uint32_t cc = c0; cc < c1; ++cc) {
        const uint32_t base = cc < n_norm ? cc * C : SBASE;
        if (RING || base + C + W > P0 + 1) need_rel = true;
    }
    // only the face rows of this group's heads are read (32-byte runs): every core reading whole REL / Q tiles
    // from the few cores that hold them was the bottleneck. Local REL layout: [GH][WRT * 32] bf16, then the sink
    // values of the group's heads.
    crel.reserve_back(WRT + 1);
    const uint32_t rbase = crel.get_write_ptr();
    auto face_off = [](uint32_t row, uint32_t f) { return (((row >> 4) * 2 + f) * 256 + (row & 15) * 16) * 2; };
    if (need_rel) {
        for (uint32_t h = 0; h < GH; ++h) {
            const uint32_t hh = g * GH + h;
            for (uint32_t t = 0; t < WRT; ++t) {
                for (uint32_t f = 0; f < 2; ++f) {
                    noc.async_read(sREL, crel, 32, {.page_id = t, .offset_bytes = face_off(hh, f)},
                                   {.offset_bytes = ((h * WRT + t) * 32 + f * 16) * 2});
                }
            }
        }
    }
    const uint32_t skoff = GH * WRT * 32 * 2;
    for (uint32_t f = 0; f < 2; ++f) {  // row 0 of the sink tile (heads in columns)
        noc.async_read(sSK, crel, 32, {.page_id = 0, .offset_bytes = face_off(0, f)},
                       {.offset_bytes = skoff + f * 32});
    }
    if (active) {
        cq.reserve_back(RT * HDT);
        const uint32_t qbase = cq.get_write_ptr();
        for (uint32_t d = 0; d < HDT; ++d) {
            volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(qbase + d * TB);
            for (uint32_t w = 0; w < TB / 4; ++w) z[w] = 0;
        }
        // q rows of the group -> rows h' of the Q tiles
        for (uint32_t h = 0; h < GH; ++h) {
            const uint32_t srow = g * GH + h;
            for (uint32_t d = 0; d < HDT; ++d) {
                for (uint32_t f = 0; f < 2; ++f) {
                    noc.async_read(sQ, cq, 32, {.page_id = d, .offset_bytes = face_off(srow, f)},
                                   {.offset_bytes = d * TB + face_off(h, f)});
                }
            }
        }
    }
    noc.async_read_barrier();
    sc.push_back(1);
    cst.push_back(2);
    if (active) cq.push_back(RT * HDT);

    volatile tt_l1_ptr uint16_t* rel = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rbase);
    volatile tt_l1_ptr uint16_t* sk = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rbase + skoff);
    CircularBuffer ck(c_K), cv(c_V), cm(c_MASK);
    for (uint32_t cc = c0; cc < c1; ++cc) {
        const uint32_t base = cc < n_norm ? cc * C : SBASE;
        ck.reserve_back(CT * HDT);
        for (uint32_t j = 0; j < CT; ++j) {
            const uint32_t t0 = (g * (BSL / 32) + (base + j * 32) / 32) * HDT;
            for (uint32_t d = 0; d < HDT; ++d) {
                noc.async_read(sK, ck, KVTB, {.page_id = t0 + d}, {.offset_bytes = (j * HDT + d) * KVTB});
            }
        }
        // mask tiles: rows < GH (the group's heads), others fully masked
        cm.reserve_back(RT * CT);
        volatile tt_l1_ptr uint16_t* m = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cm.get_write_ptr());
        for (uint32_t j = 0; j < CT; ++j) {
            volatile tt_l1_ptr uint32_t* m32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(m + j * 1024);
            for (uint32_t i = 0; i < 512; ++i) m32[i] = ((uint32_t)NEG << 16) | NEG;
            volatile tt_l1_ptr uint16_t* mt = m + j * 1024;
            const uint32_t k0 = base + j * 32;
            for (uint32_t col = 0; col < 32; ++col) {
                const uint32_t key = k0 + col;
                int32_t dist = -1;  // -1: masked; -2: sink; else the distance
                if (key == SINK) {
                    dist = -2;
                } else if (RING) {
                    if (key < R) {
                        const uint32_t dd = cpos >= key ? cpos - key : cpos + R - key;
                        if (dd <= dmax) dist = (int32_t)dd;
                    }
                } else if (key <= P0) {
                    dist = (int32_t)(P0 - key);
                }
                if (dist == -1) continue;
                for (uint32_t h = 0; h < GH; ++h) {
                    uint16_t v;
                    if (dist == -2) {
                        const uint32_t hh = g * GH + h;
                        v = sk[(hh >> 4) * 16 + (hh & 15)];
                    } else if ((uint32_t)dist < WRT * 32 && (RING || (uint32_t)dist < W)) {
                        v = rel[h * WRT * 32 + (uint32_t)dist];
                    } else {
                        v = 0;
                    }
                    mt[elem_off(h, col)] = v;
                }
            }
        }
        noc.async_read_barrier();
        ck.push_back(CT * HDT);
        cm.push_back(RT * CT);
        cv.reserve_back(CT * HDT);
        for (uint32_t j = 0; j < CT; ++j) {
            const uint32_t t0 = (g * (BSL / 32) + (base + j * 32) / 32) * HDT;
            for (uint32_t d = 0; d < HDT; ++d) {
                noc.async_read(sV, cv, KVTB, {.page_id = t0 + d}, {.offset_bytes = (j * HDT + d) * KVTB});
            }
        }
        noc.async_read_barrier();
        cv.push_back(CT * HDT);
    }
}

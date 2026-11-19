// GQA split-K attention (flash-decoding) over a paged bf16 KV cache — reader.
// Core (g, s): kv head g, split s of the key range [0, N) with N = POS[0] + T.
// Q rows of the group are folded: row r = h' * T + t (h' < GH q heads of the group, t < T tokens), RT row tiles.
// Keys are processed in chunks of C = CT*32 positions; split s takes chunks [s*nch/S, (s+1)*nch/S).
//   Q   : pre-scaled by 1/sqrt(HD).  QMODE 0: [1, T, NH(pad 32), HD] bf16 TILE (page t*HDT + d), gathered here;
//         QMODE 1 (T == 32): [1, NH, 32, HD] (page h*HDT + d), row tile h' of the group = head g*GH + h'
//   SC  : bf16 tile, ones in row 0 of each face (reduce scaler);  CF : 2 fp32 tiles (-1e30, 0)
//   Kc/Vc : [blocks, NKV, BS, HD] bf16 TILE;  PT : int32 row-major [1, nblocks] page table (logical -> physical block)
//   POS : int32 [T] (positions of the T rows; only POS[0] is used, rows are consecutive)
// CT: 0 T, 1 RT, 2 CT, 3 S, 4 BS, 5 NKV, 6 POSB, 7 GH, 8 HDT, 9 QMODE, 10 PTB, then accessors Q, Kc, Vc, POS, SC, CF, PT
// RT: 0..6 addrs, 7 g, 8 s
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t c_Q = 0, c_K = 1, c_V = 2, c_MASK = 3, c_SC = 4, c_INFO = 5, c_CONST = 19, c_SCR = 20, c_PT = 21;

FORCE_INLINE uint32_t elem_off(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

void kernel_main() {
    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t RT = get_compile_time_arg_val(1);
    constexpr uint32_t CT = get_compile_time_arg_val(2);
    constexpr uint32_t S = get_compile_time_arg_val(3);
    constexpr uint32_t BS = get_compile_time_arg_val(4);
    constexpr uint32_t NKV = get_compile_time_arg_val(5);
    constexpr uint32_t POSB = get_compile_time_arg_val(6);
    constexpr uint32_t GH = get_compile_time_arg_val(7);
    constexpr uint32_t HDT = get_compile_time_arg_val(8);
    constexpr uint32_t QMODE = get_compile_time_arg_val(9);
    constexpr uint32_t PTB = get_compile_time_arg_val(10);
    constexpr auto aQ = TensorAccessorArgs<11>();
    constexpr auto aK = TensorAccessorArgs<aQ.next_compile_time_args_offset()>();
    constexpr auto aV = TensorAccessorArgs<aK.next_compile_time_args_offset()>();
    constexpr auto aP = TensorAccessorArgs<aV.next_compile_time_args_offset()>();
    constexpr auto aSC = TensorAccessorArgs<aP.next_compile_time_args_offset()>();
    constexpr auto aCF = TensorAccessorArgs<aSC.next_compile_time_args_offset()>();
    constexpr auto aPT = TensorAccessorArgs<aCF.next_compile_time_args_offset()>();
    const auto sQ = TensorAccessor(aQ, get_arg_val<uint32_t>(0));
    const auto sK = TensorAccessor(aK, get_arg_val<uint32_t>(1));
    const auto sV = TensorAccessor(aV, get_arg_val<uint32_t>(2));
    const auto sP = TensorAccessor(aP, get_arg_val<uint32_t>(3));
    const auto sSC = TensorAccessor(aSC, get_arg_val<uint32_t>(4));
    const auto sCF = TensorAccessor(aCF, get_arg_val<uint32_t>(5));
    const auto sPT = TensorAccessor(aPT, get_arg_val<uint32_t>(6));
    const uint32_t g = get_arg_val<uint32_t>(7);
    const uint32_t s = get_arg_val<uint32_t>(8);

    constexpr uint32_t TB = 2048;  // bf16 tile bytes
    constexpr uint32_t C = CT * 32;
    constexpr uint16_t NEG = 0xC6EA;  // bf16 -29952: masked logit
    Noc noc;

    // ---- positions -> this split's chunk range ----
    CircularBuffer scr(c_SCR);
    scr.reserve_back(HDT);
    const uint32_t sbase = scr.get_write_ptr();
    noc.async_read(sP, scr, POSB, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    const uint32_t P0 = (uint32_t)(reinterpret_cast<volatile tt_l1_ptr int32_t*>(sbase)[0]);
    const uint32_t N = P0 + T;
    const uint32_t nch = (N + C - 1) / C;
    // splits 0..min(S, nch)-1 are non-empty (the combine reads only those)
    const uint32_t base_n = nch / S, rem = nch % S;
    const uint32_t c0 = s * base_n + (s < rem ? s : rem);
    const uint32_t c1 = c0 + base_n + (s < rem ? 1 : 0);

    CircularBuffer info(c_INFO);
    info.reserve_back(1);
    {
        volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(info.get_write_ptr());
        w[0] = c1 - c0;
        w[1] = c0;
        w[2] = P0;
    }
    info.push_back(1);

    // ---- reduce scaler and constants ----
    CircularBuffer sc(c_SC), cst(c_CONST), cq(c_Q);
    sc.reserve_back(1);
    noc.async_read(sSC, sc, TB, {.page_id = 0}, {.offset_bytes = 0});
    cst.reserve_back(2);
    noc.async_read(sCF, cst, 4096, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read(sCF, cst, 4096, {.page_id = 1}, {.offset_bytes = 4096});
    const bool active = c1 > c0;  // idle splits skip Q entirely (compute does not wait for it)
    if (active) cq.reserve_back(RT * HDT);
    if (!active) {
        noc.async_read_barrier();
    } else if constexpr (QMODE == 1) {
        for (uint32_t r = 0; r < RT; ++r) {
            for (uint32_t d = 0; d < HDT; ++d) {
                noc.async_read(sQ, cq, TB, {.page_id = (g * GH + r) * HDT + d}, {.offset_bytes = (r * HDT + d) * TB});
            }
        }
        noc.async_read_barrier();
    } else {
        // gather the group's GH heads x T tokens: row r = h' * T + t (rows >= GH*T left as is: independent rows)
        noc.async_read_barrier();
        const uint32_t qbase = cq.get_write_ptr();
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t d = 0; d < HDT; ++d) {
                noc.async_read(sQ, scr, TB, {.page_id = t * HDT + d}, {.offset_bytes = d * TB});
            }
            noc.async_read_barrier();
            for (uint32_t h = 0; h < GH; ++h) {
                const uint32_t r = h * T + t;
                const uint32_t rt = r >> 5, rr = r & 31;
                const uint32_t srow = g * GH + h;
                for (uint32_t d = 0; d < HDT; ++d) {
                    for (uint32_t f = 0; f < 2; ++f) {
                        volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                            qbase + (rt * HDT + d) * TB + (((rr >> 4) * 2 + f) * 256 + (rr & 15) * 16) * 2);
                        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                            sbase + d * TB + (((srow >> 4) * 2 + f) * 256 + (srow & 15) * 16) * 2);
                        for (uint32_t w = 0; w < 8; ++w) dst[w] = src[w];
                    }
                }
            }
        }
    }
    sc.push_back(1);
    cst.push_back(2);
    if (active) cq.push_back(RT * HDT);

    // ---- K/V chunks (+ causal mask tiles for chunks reaching past P0) ----
    CircularBuffer ck(c_K), cv(c_V), cm(c_MASK), cpt(c_PT);
    cpt.reserve_back(1);
    noc.async_read(sPT, cpt, PTB, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    volatile tt_l1_ptr int32_t* PT = reinterpret_cast<volatile tt_l1_ptr int32_t*>(cpt.get_write_ptr());
    for (uint32_t c = c0; c < c1; ++c) {
        const uint32_t base = c * C;
        ck.reserve_back(CT * HDT);
        for (uint32_t j = 0; j < CT; ++j) {
            const uint32_t p = base + j * 32;
            const uint32_t b = (uint32_t)PT[p / BS], rt = (p % BS) / 32;
            const uint32_t t0 = ((b * NKV + g) * (BS / 32) + rt) * HDT;
            for (uint32_t d = 0; d < HDT; ++d) {
                noc.async_read(sK, ck, TB, {.page_id = t0 + d}, {.offset_bytes = (j * HDT + d) * TB});
            }
        }
        noc.async_read_barrier();
        ck.push_back(CT * HDT);
        if (base + C > P0 + 1) {
            cm.reserve_back(RT * CT);
            volatile tt_l1_ptr uint16_t* m = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cm.get_write_ptr());
            for (uint32_t rt = 0; rt < RT; ++rt) {
                for (uint32_t rr = 0; rr < 32; ++rr) {
                    const uint32_t r = rt * 32 + rr;
                    const uint32_t lim = (r < GH * T) ? P0 + (r % T) : N - 1;
                    for (uint32_t j = 0; j < CT; ++j) {
                        volatile tt_l1_ptr uint16_t* mt = m + (rt * CT + j) * 1024;
                        const uint32_t kb = base + j * 32;
                        for (uint32_t col = 0; col < 32; ++col) {
                            mt[elem_off(rr, col)] = (kb + col <= lim) ? (uint16_t)0 : NEG;
                        }
                    }
                }
            }
            cm.push_back(RT * CT);
        }
        cv.reserve_back(CT * HDT);
        for (uint32_t j = 0; j < CT; ++j) {
            const uint32_t p = base + j * 32;
            const uint32_t b = (uint32_t)PT[p / BS], rt = (p % BS) / 32;
            const uint32_t t0 = ((b * NKV + g) * (BS / 32) + rt) * HDT;
            for (uint32_t d = 0; d < HDT; ++d) {
                noc.async_read(sV, cv, TB, {.page_id = t0 + d}, {.offset_bytes = (j * HDT + d) * TB});
            }
        }
        noc.async_read_barrier();
        cv.push_back(CT * HDT);
    }
}

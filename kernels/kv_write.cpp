// Write T token rows of K and V into a paged cache [blocks, NKV, BS, HD] (bf16 or bfp8_b); logical block b -> physical PT[b].
// One core per (kv head h, column tile c); it walks the T rows in order and does a read-modify-write
// of the cache tile, so rows of one step that share a tile never race.
//   src  : [1, T, NKV(pad 32), HD] bf16 TILE (tile (t, c) holds heads in rows, HD/32 column tiles)
//   POS  : int32 row-major [T] positions
//   PT   : int32 row-major [1, nblocks] page table (one page)
// Lanes (batched decode): NU sequences of T rows each (lane u = source rows u*T..), each with its own POS and PT.
// CT: 0 T (rows per lane), 1 NKV, 2 HDT (HD/32), 3 BS (block size, multiple of 32), 4 pos_bytes, 5 PT bytes,
//     6 BFP8 (cache format), 7 NU (lanes), then accessors Ksrc, Vsrc, Kc, Vc, POS, PT
// bfp8_b tile (1088 B): 64 exponent bytes (one per face row: face f = (row/16)*2 + col/16, index f*16 + row%16), then
// 4 faces x 16 rows x 16 mantissa bytes (sign << 7 | 7-bit magnitude incl. the hidden bit, shifted to the row's
// shared exponent = max exponent of its 16 values; round to nearest even, as tt-metal's host packer). A cache row
// (one position) owns its own exponents, so rewriting it never touches the other rows of the tile.
// RT: 0..3 addrs Ksrc, Vsrc, Kc, Vc; then per lane (POS addr, PT addr); then h, c

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t c_SCR = 0;  // 3 tiles: [pos][src][dst]
constexpr uint32_t c_PT = 1;   // page table

FORCE_INLINE uint32_t elem_off(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

// 16 bf16 values -> shared exponent + 16 bfp8_b mantissa bytes
FORCE_INLINE void to_bfp8(const uint16_t* v, volatile tt_l1_ptr uint8_t* exp_out, volatile tt_l1_ptr uint8_t* man_out) {
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
        if (e != 0) {  // zero / denormal -> 0
            uint32_t m = (1u << 23) | ((b & 0x7f) << 16);
            const uint32_t d = se - e;
            m = d > 31 ? 0 : (m >> d);
            const uint32_t rnd = m & 0x1ffff;  // 17 bits below the 7 kept
            m >>= 17;
            if (rnd > 0x10000 || (rnd == 0x10000 && (m & 1))) {
                m += 1;
            }
            if (m > 127) {
                m = 127;
            }
            out = m ? (((b >> 15) & 1) << 7) | m : 0;
        }
        man_out[i] = (uint8_t)out;
    }
}

void kernel_main() {
    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t NKV = get_compile_time_arg_val(1);
    constexpr uint32_t HDT = get_compile_time_arg_val(2);
    constexpr uint32_t BS = get_compile_time_arg_val(3);
    constexpr uint32_t POSB = get_compile_time_arg_val(4);
    constexpr uint32_t PTB = get_compile_time_arg_val(5);
    constexpr uint32_t BFP8 = get_compile_time_arg_val(6);
    constexpr uint32_t NU = get_compile_time_arg_val(7);
    constexpr auto aKs = TensorAccessorArgs<8>();
    constexpr auto aVs = TensorAccessorArgs<aKs.next_compile_time_args_offset()>();
    constexpr auto aKc = TensorAccessorArgs<aVs.next_compile_time_args_offset()>();
    constexpr auto aVc = TensorAccessorArgs<aKc.next_compile_time_args_offset()>();
    constexpr auto aP = TensorAccessorArgs<aVc.next_compile_time_args_offset()>();
    constexpr auto aPT = TensorAccessorArgs<aP.next_compile_time_args_offset()>();
    const auto sKs = TensorAccessor(aKs, get_arg_val<uint32_t>(0));
    const auto sVs = TensorAccessor(aVs, get_arg_val<uint32_t>(1));
    const auto sKc = TensorAccessor(aKc, get_arg_val<uint32_t>(2));
    const auto sVc = TensorAccessor(aVc, get_arg_val<uint32_t>(3));
    const uint32_t h = get_arg_val<uint32_t>(4 + 2 * NU);
    const uint32_t c = get_arg_val<uint32_t>(5 + 2 * NU);

    constexpr uint32_t TB = 2048;            // bf16 tile bytes (source rows, scratch slots)
    constexpr uint32_t CB_ = BFP8 ? 1088 : 2048;  // cache tile bytes
    Noc noc;
    CircularBuffer scr(c_SCR);
    scr.reserve_back(3);
    const uint32_t base = scr.get_write_ptr();
    CircularBuffer cpt(c_PT);
    cpt.reserve_back(1);
  for (uint32_t u = 0; u < NU; ++u) {
    const auto sP = TensorAccessor(aP, get_arg_val<uint32_t>(4 + 2 * u));
    const auto sPT = TensorAccessor(aPT, get_arg_val<uint32_t>(5 + 2 * u));
    const uint32_t R0 = u * T;
    noc.async_read(sP, scr, POSB, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read(sPT, cpt, PTB, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    volatile tt_l1_ptr int32_t* PT = reinterpret_cast<volatile tt_l1_ptr int32_t*>(cpt.get_write_ptr());
    volatile tt_l1_ptr int32_t* pos = reinterpret_cast<volatile tt_l1_ptr int32_t*>(base);
    int32_t P[T];
    for (uint32_t t = 0; t < T; ++t) {
        P[t] = pos[t];
    }
    volatile tt_l1_ptr uint16_t* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + TB);
    volatile tt_l1_ptr uint16_t* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + 2 * TB);

    for (uint32_t kv = 0; kv < 2; ++kv) {
        int32_t cur = -1;
        for (uint32_t t = 0; t < T; ++t) {
            const uint32_t p = (uint32_t)P[t];
            const uint32_t b = (uint32_t)PT[p / BS], r = p % BS;
            const int32_t tile_id = (int32_t)(((b * NKV + h) * (BS / 32) + r / 32) * HDT + c);
            if (tile_id != cur) {
                if (cur >= 0) {
                    if (kv == 0) {
                        noc.async_write(scr, sKc, CB_, {.offset_bytes = 2 * TB}, {.page_id = (uint32_t)cur});
                    } else {
                        noc.async_write(scr, sVc, CB_, {.offset_bytes = 2 * TB}, {.page_id = (uint32_t)cur});
                    }
                    noc.async_write_barrier();
                }
                if (kv == 0) {
                    noc.async_read(sKc, scr, CB_, {.page_id = (uint32_t)tile_id}, {.offset_bytes = 2 * TB});
                } else {
                    noc.async_read(sVc, scr, CB_, {.page_id = (uint32_t)tile_id}, {.offset_bytes = 2 * TB});
                }
                cur = tile_id;
            }
            if (kv == 0) {
                noc.async_read(sKs, scr, TB, {.page_id = (R0 + t) * HDT + c}, {.offset_bytes = TB});
            } else {
                noc.async_read(sVs, scr, TB, {.page_id = (R0 + t) * HDT + c}, {.offset_bytes = TB});
            }
            noc.async_read_barrier();
            const uint32_t rr = r % 32;
            if constexpr (BFP8) {
                volatile tt_l1_ptr uint8_t* d8 = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(dst);
                for (uint32_t half = 0; half < 2; ++half) {
                    uint16_t v[16];
                    for (uint32_t i = 0; i < 16; ++i) {
                        v[i] = src[elem_off(h, half * 16 + i)];
                    }
                    const uint32_t f = (rr >> 4) * 2 + half;
                    to_bfp8(v, d8 + f * 16 + (rr & 15), d8 + 64 + f * 256 + (rr & 15) * 16);
                }
            } else {
                for (uint32_t col = 0; col < 32; ++col) {
                    dst[elem_off(rr, col)] = src[elem_off(h, col)];
                }
            }
        }
        if (cur >= 0) {
            if (kv == 0) {
                noc.async_write(scr, sKc, CB_, {.offset_bytes = 2 * TB}, {.page_id = (uint32_t)cur});
            } else {
                noc.async_write(scr, sVc, CB_, {.offset_bytes = 2 * TB}, {.page_id = (uint32_t)cur});
            }
            noc.async_write_barrier();
        }
    }
  }  // lanes
}

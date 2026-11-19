// Write T token rows of K and V into a paged bf16 cache [blocks, NKV, BS, HD] (identity page table).
// One core per (kv head h, column tile c); it walks the T rows in order and does a read-modify-write
// of the cache tile, so rows of one step that share a tile never race.
//   src  : [1, T, NKV(pad 32), HD] bf16 TILE (tile (t, c) holds heads in rows, HD/32 column tiles)
//   POS  : int32 row-major [T] positions
// CT: 0 T, 1 NKV, 2 HDT (HD/32), 3 BS (block size, multiple of 32), 4 pos_bytes, then accessors Ksrc, Vsrc, Kc, Vc, POS
// RT: 0..4 addrs, 5 h, 6 c

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t c_SCR = 0;  // 3 tiles: [pos][src][dst]

FORCE_INLINE uint32_t elem_off(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

void kernel_main() {
    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t NKV = get_compile_time_arg_val(1);
    constexpr uint32_t HDT = get_compile_time_arg_val(2);
    constexpr uint32_t BS = get_compile_time_arg_val(3);
    constexpr uint32_t POSB = get_compile_time_arg_val(4);
    constexpr auto aKs = TensorAccessorArgs<5>();
    constexpr auto aVs = TensorAccessorArgs<aKs.next_compile_time_args_offset()>();
    constexpr auto aKc = TensorAccessorArgs<aVs.next_compile_time_args_offset()>();
    constexpr auto aVc = TensorAccessorArgs<aKc.next_compile_time_args_offset()>();
    constexpr auto aP = TensorAccessorArgs<aVc.next_compile_time_args_offset()>();
    const auto sKs = TensorAccessor(aKs, get_arg_val<uint32_t>(0));
    const auto sVs = TensorAccessor(aVs, get_arg_val<uint32_t>(1));
    const auto sKc = TensorAccessor(aKc, get_arg_val<uint32_t>(2));
    const auto sVc = TensorAccessor(aVc, get_arg_val<uint32_t>(3));
    const auto sP = TensorAccessor(aP, get_arg_val<uint32_t>(4));
    const uint32_t h = get_arg_val<uint32_t>(5);
    const uint32_t c = get_arg_val<uint32_t>(6);

    constexpr uint32_t TB = 2048;  // bf16 tile bytes
    Noc noc;
    CircularBuffer scr(c_SCR);
    scr.reserve_back(3);
    const uint32_t base = scr.get_write_ptr();
    noc.async_read(sP, scr, POSB, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
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
            const uint32_t b = p / BS, r = p % BS;
            const int32_t tile_id = (int32_t)(((b * NKV + h) * (BS / 32) + r / 32) * HDT + c);
            if (tile_id != cur) {
                if (cur >= 0) {
                    if (kv == 0) {
                        noc.async_write(scr, sKc, TB, {.offset_bytes = 2 * TB}, {.page_id = (uint32_t)cur});
                    } else {
                        noc.async_write(scr, sVc, TB, {.offset_bytes = 2 * TB}, {.page_id = (uint32_t)cur});
                    }
                    noc.async_write_barrier();
                }
                if (kv == 0) {
                    noc.async_read(sKc, scr, TB, {.page_id = (uint32_t)tile_id}, {.offset_bytes = 2 * TB});
                } else {
                    noc.async_read(sVc, scr, TB, {.page_id = (uint32_t)tile_id}, {.offset_bytes = 2 * TB});
                }
                cur = tile_id;
            }
            if (kv == 0) {
                noc.async_read(sKs, scr, TB, {.page_id = t * HDT + c}, {.offset_bytes = TB});
            } else {
                noc.async_read(sVs, scr, TB, {.page_id = t * HDT + c}, {.offset_bytes = TB});
            }
            noc.async_read_barrier();
            const uint32_t rr = r % 32;
            for (uint32_t col = 0; col < 32; ++col) {
                dst[elem_off(rr, col)] = src[elem_off(h, col)];
            }
        }
        if (cur >= 0) {
            if (kv == 0) {
                noc.async_write(scr, sKc, TB, {.offset_bytes = 2 * TB}, {.page_id = (uint32_t)cur});
            } else {
                noc.async_write(scr, sVc, TB, {.offset_bytes = 2 * TB}, {.page_id = (uint32_t)cur});
            }
            noc.async_write_barrier();
        }
    }
}

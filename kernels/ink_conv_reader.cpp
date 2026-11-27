// Inkling decode: short conv + residual of one token (row 0), split over column tiles — reader.
// Core: tiles j0..j0+NT-1 of X (row 0 = the token), HIST (rows 1..3 = lags 1..3), TAPS (rows 0..3), B (residual,
// optional), and the 0/1 matrices E0 (row 0 -> row 0), ONES0 (row 0 = sum of rows 0..3), SHIFT (row r -> r + 1).
// CB 0 X, 1 HIST, 2 TAPS, 3 B, 4 consts (E0, ONES0, SHIFT), 7 scratch.  CT: accessors X (L1), HIST (DRAM),
// TAPS (DRAM), B (L1), C (DRAM), WCOL (L1).  RT: 0 x, 1 hist, 2 taps, 3 b, 4 consts, 5 j0, 6 NT, 7 use_b,
// 8 moe (K experts, 0 = off), 9 wcol, 10 column tiles of DN (C / 32)
// MoE mode (the post-MLP conv): X = DN [1, K, 32, C] (row 0 of each expert slot): tile j of X is built dense,
// row k = row 0 of DN tile (k, j0 + j), and E0 is replaced by W (row 0, col k = routing weight k, from row 0 of
// WCOL tile k), so that "E0 X" = sum_k w_k DN_k: the combine of the experts costs no extra program.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr auto aX = TensorAccessorArgs<0>();
    constexpr auto aH = TensorAccessorArgs<aX.next_compile_time_args_offset()>();
    constexpr auto aT = TensorAccessorArgs<aH.next_compile_time_args_offset()>();
    constexpr auto aB = TensorAccessorArgs<aT.next_compile_time_args_offset()>();
    constexpr auto aC = TensorAccessorArgs<aB.next_compile_time_args_offset()>();
    constexpr auto aW = TensorAccessorArgs<aC.next_compile_time_args_offset()>();
    const auto sX = TensorAccessor(aX, get_arg_val<uint32_t>(0));
    const auto sH = TensorAccessor(aH, get_arg_val<uint32_t>(1));
    const auto sT = TensorAccessor(aT, get_arg_val<uint32_t>(2));
    const auto sB = TensorAccessor(aB, get_arg_val<uint32_t>(3));
    const auto sC = TensorAccessor(aC, get_arg_val<uint32_t>(4));
    const uint32_t j0 = get_arg_val<uint32_t>(5), NT = get_arg_val<uint32_t>(6), use_b = get_arg_val<uint32_t>(7);
    const uint32_t moe = get_arg_val<uint32_t>(8);
    constexpr uint32_t TB = 2048;
    Noc noc;
    CircularBuffer cx(0), ch(1), ct(2), cb(3), cc(4), cs(7);
    cc.reserve_back(3);
    cx.reserve_back(NT);
    if (moe) {
        const auto sW = TensorAccessor(aW, get_arg_val<uint32_t>(9));
        for (uint32_t i = 1; i < 3; ++i) noc.async_read(sC, cc, TB, {.page_id = i}, {.offset_bytes = i * TB});
        cs.reserve_back(1);
        for (uint32_t k = 0; k < moe; ++k) noc.async_read(sW, cs, 32, {.page_id = k}, {.offset_bytes = k * 32});
        volatile tt_l1_ptr uint32_t* zx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cx.get_write_ptr());
        volatile tt_l1_ptr uint32_t* zc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cc.get_write_ptr());
        for (uint32_t i = 0; i < NT * 512; ++i) zx[i] = 0;
        for (uint32_t i = 0; i < 512; ++i) zc[i] = 0;
        for (uint32_t j = 0; j < NT; ++j)
            for (uint32_t k = 0; k < moe; ++k)
                for (uint32_t h = 0; h < 2; ++h)
                    noc.async_read(sX, cx, 32, {.page_id = k * get_arg_val<uint32_t>(10) + j0 + j,
                                                .offset_bytes = h * 512},
                                   {.offset_bytes = j * TB + k * 32 + h * 512});
        noc.async_read_barrier();
        volatile tt_l1_ptr uint16_t* sw = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cs.get_write_ptr());
        volatile tt_l1_ptr uint16_t* w0 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cc.get_write_ptr());
        for (uint32_t k = 0; k < moe; ++k) w0[k] = sw[k * 16];  // row 0, col k (face 0)
    } else {
        for (uint32_t i = 0; i < 3; ++i) noc.async_read(sC, cc, TB, {.page_id = i}, {.offset_bytes = i * TB});
    }
    ch.reserve_back(NT);
    ct.reserve_back(NT);
    for (uint32_t j = 0; j < NT; ++j) {
        if (!moe) noc.async_read(sX, cx, TB, {.page_id = j0 + j}, {.offset_bytes = j * TB});
        noc.async_read(sH, ch, TB, {.page_id = j0 + j}, {.offset_bytes = j * TB});
        noc.async_read(sT, ct, TB, {.page_id = j0 + j}, {.offset_bytes = j * TB});
    }
    if (use_b) {
        cb.reserve_back(NT);
        for (uint32_t j = 0; j < NT; ++j) noc.async_read(sB, cb, TB, {.page_id = j0 + j}, {.offset_bytes = j * TB});
    }
    noc.async_read_barrier();
    cc.push_back(3);
    cx.push_back(NT);
    ch.push_back(NT);
    ct.push_back(NT);
    if (use_b) cb.push_back(NT);
}

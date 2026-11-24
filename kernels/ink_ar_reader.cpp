// Inkling decode: block attention residual + rms norm of one token (see tsugumi/inkling/ares_kernel.py) — reader.
// Core: column tiles j0..j0+NT-1 of X (row 0 = the raw state h), NB (rows 0..nb-1 = normalized block states),
// QT (rows 31, 30, .. = the NQ pre-scaled queries), the 0/1 tile SH[nb] (row 0 -> row nb). Phase 2 also: the NC1
// float32 Gram partials of phase 1 and the NQ norm weight tiles (all rows = the weights) of the core's columns.
// CB 0 X, 1 NB, 2 QT, 3 SH, 4 P (phase 2, streamed), 6 LN (phase 2, streamed in (tile, query) order).
// CT: accessors X (L1), NB (L1), QT (DRAM), C (DRAM), P (L1), LN (DRAM).
// RT: 0 x, 1 nb tensor, 2 qt, 3 consts, 4 part, 5..7 ln0..2, 8 j0, 9 NT, 10 use_nb, 11 nb, 12 phase, 13 NC1, 14 NQ,
//     15 use_ln
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr auto aX = TensorAccessorArgs<0>();
    constexpr auto aN = TensorAccessorArgs<aX.next_compile_time_args_offset()>();
    constexpr auto aQ = TensorAccessorArgs<aN.next_compile_time_args_offset()>();
    constexpr auto aC = TensorAccessorArgs<aQ.next_compile_time_args_offset()>();
    constexpr auto aP = TensorAccessorArgs<aC.next_compile_time_args_offset()>();
    constexpr auto aL = TensorAccessorArgs<aP.next_compile_time_args_offset()>();
    const auto sX = TensorAccessor(aX, get_arg_val<uint32_t>(0));
    const auto sN = TensorAccessor(aN, get_arg_val<uint32_t>(1));
    const auto sQ = TensorAccessor(aQ, get_arg_val<uint32_t>(2));
    const auto sC = TensorAccessor(aC, get_arg_val<uint32_t>(3));
    const auto sP = TensorAccessor(aP, get_arg_val<uint32_t>(4));
    const uint32_t j0 = get_arg_val<uint32_t>(8), NT = get_arg_val<uint32_t>(9), use_nb = get_arg_val<uint32_t>(10);
    const uint32_t nb = get_arg_val<uint32_t>(11), phase = get_arg_val<uint32_t>(12), NC1 = get_arg_val<uint32_t>(13);
    const uint32_t NQ = get_arg_val<uint32_t>(14), use_ln = get_arg_val<uint32_t>(15);
    constexpr uint32_t TB = 2048;
    Noc noc;
    CircularBuffer cx(0), cn(1), cq(2), cs(3), cp(4), cl(6);
    if (phase == 2) {  // the partials first: the compute kernel sums them while the rest arrives
        for (uint32_t i = 0; i < NC1; ++i) {  // one at a time (CB of 2)
            cp.reserve_back(1);
            noc.async_read(sP, cp, 4096, {.page_id = i}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cp.push_back(1);
        }
    }
    cs.reserve_back(1);
    noc.async_read(sC, cs, TB, {.page_id = nb}, {.offset_bytes = 0});
    cx.reserve_back(NT);
    cq.reserve_back(NT);
    for (uint32_t j = 0; j < NT; ++j) {
        noc.async_read(sX, cx, TB, {.page_id = j0 + j}, {.offset_bytes = j * TB});
        noc.async_read(sQ, cq, TB, {.page_id = j0 + j}, {.offset_bytes = j * TB});
    }
    if (use_nb) {
        cn.reserve_back(NT);
        for (uint32_t j = 0; j < NT; ++j) noc.async_read(sN, cn, TB, {.page_id = j0 + j}, {.offset_bytes = j * TB});
    }
    noc.async_read_barrier();
    cs.push_back(1);
    cx.push_back(NT);
    cq.push_back(NT);
    if (use_nb) cn.push_back(NT);
    if (phase == 2 && use_ln) {
        for (uint32_t j = 0; j < NT; ++j) {  // in the order of use (tile j, query k), one at a time
            for (uint32_t k = 0; k < NQ; ++k) {
                const auto sL = TensorAccessor(aL, get_arg_val<uint32_t>(5 + k));
                cl.reserve_back(1);
                noc.async_read(sL, cl, TB, {.page_id = j0 + j}, {.offset_bytes = 0});
                noc.async_read_barrier();
                cl.push_back(1);
            }
        }
    }
}

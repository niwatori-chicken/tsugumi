// Causal depthwise conv (kernel 4) + SiLU over T rows — reader.
// Unit u = (segment s, column tile c): output row tiles [s*RPS, min((s+1)*RPS, NT)) of column tile c.
//   X     : [1,1,T,C] bf16 TILE (conv input, T = NT*32 rows)
//   CARRY : [1,1,32,C] bf16 TILE, rows 29..31 = the 3 input rows before row 0 (oldest first)
//   TAPj  : [1,1,32,C] bf16 TILE, row 0 = tap j (multiplies input row t-3+j)
//   PQ    : [1,1,256,32] bf16 TILE, 8 shift tiles P0..P3, Q0..Q3
// CT: 0 NT, 1 CTL (C/32), 2 RPS, 3 XW (row stride of X in tiles), then accessors X, CARRY, TAP0..3, PQ.  RT: 0..6 addrs, 7 u0, 8 u1
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t c_X = 0, c_TAP = 1, c_PQ = 2;

void kernel_main() {
    constexpr uint32_t NT = get_compile_time_arg_val(0);
    constexpr uint32_t CTL = get_compile_time_arg_val(1);
    constexpr uint32_t RPS = get_compile_time_arg_val(2);
    constexpr uint32_t XW = get_compile_time_arg_val(3);
    constexpr auto aX = TensorAccessorArgs<4>();
    constexpr auto aC = TensorAccessorArgs<aX.next_compile_time_args_offset()>();
    constexpr auto aT0 = TensorAccessorArgs<aC.next_compile_time_args_offset()>();
    constexpr auto aT1 = TensorAccessorArgs<aT0.next_compile_time_args_offset()>();
    constexpr auto aT2 = TensorAccessorArgs<aT1.next_compile_time_args_offset()>();
    constexpr auto aT3 = TensorAccessorArgs<aT2.next_compile_time_args_offset()>();
    constexpr auto aPQ = TensorAccessorArgs<aT3.next_compile_time_args_offset()>();
    const auto sX = TensorAccessor(aX, get_arg_val<uint32_t>(0));
    const auto sC = TensorAccessor(aC, get_arg_val<uint32_t>(1));
    const auto sT0 = TensorAccessor(aT0, get_arg_val<uint32_t>(2));
    const auto sT1 = TensorAccessor(aT1, get_arg_val<uint32_t>(3));
    const auto sT2 = TensorAccessor(aT2, get_arg_val<uint32_t>(4));
    const auto sT3 = TensorAccessor(aT3, get_arg_val<uint32_t>(5));
    const auto sPQ = TensorAccessor(aPQ, get_arg_val<uint32_t>(6));
    const uint32_t u0 = get_arg_val<uint32_t>(7);
    const uint32_t u1 = get_arg_val<uint32_t>(8);
    constexpr uint32_t TB = 2048;
    Noc noc;
    CircularBuffer cx(c_X), ct(c_TAP), cpq(c_PQ);
    cpq.reserve_back(8);
    for (uint32_t k = 0; k < 8; ++k) {
        noc.async_read(sPQ, cpq, TB, {.page_id = k}, {.offset_bytes = k * TB});
    }
    noc.async_read_barrier();
    cpq.push_back(8);
    for (uint32_t u = u0; u < u1; ++u) {
        const uint32_t s = u / CTL, c = u % CTL;
        const uint32_t i0 = s * RPS, i1 = (i0 + RPS < NT) ? i0 + RPS : NT;
        ct.reserve_back(4);
        noc.async_read(sT0, ct, TB, {.page_id = c}, {.offset_bytes = 0});
        noc.async_read(sT1, ct, TB, {.page_id = c}, {.offset_bytes = TB});
        noc.async_read(sT2, ct, TB, {.page_id = c}, {.offset_bytes = 2 * TB});
        noc.async_read(sT3, ct, TB, {.page_id = c}, {.offset_bytes = 3 * TB});
        cx.reserve_back(1);
        if (i0 == 0) {
            noc.async_read(sC, cx, TB, {.page_id = c}, {.offset_bytes = 0});
        } else {
            noc.async_read(sX, cx, TB, {.page_id = (i0 - 1) * XW + c}, {.offset_bytes = 0});
        }
        noc.async_read_barrier();
        ct.push_back(4);
        cx.push_back(1);
        for (uint32_t i = i0; i < i1; ++i) {
            cx.reserve_back(1);
            noc.async_read(sX, cx, TB, {.page_id = i * XW + c}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cx.push_back(1);
        }
    }
}

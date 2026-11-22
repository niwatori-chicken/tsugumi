// GQA split-K attention — combine writer: bf16 out tiles (row r' = h' * T + t of row tile r) into
// OUT [1, 1, 32, NH * HD] (row t, columns of head g*GH + h').
// T == 32: tile copy (row tile r = head h').  32 % T == 0, T < 32: rows are regrouped per head in L1.
// CT: 0 T, 1 GH, 2 HDT, 3 DT, then accessor OUT.  RT: 0 addr, 1 g, 2 r, 3 dg (value tiles dg*DT..), 4 R0 (lane's first OUT row; T < 32)
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t c_OUT = 8;

FORCE_INLINE uint32_t elem_off(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

void kernel_main() {
    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t GH = get_compile_time_arg_val(1);
    constexpr uint32_t HDT = get_compile_time_arg_val(2);
    constexpr uint32_t DT = get_compile_time_arg_val(3);
    constexpr auto aO = TensorAccessorArgs<4>();
    const auto sO = TensorAccessor(aO, get_arg_val<uint32_t>(0));
    const uint32_t g = get_arg_val<uint32_t>(1);
    const uint32_t r = get_arg_val<uint32_t>(2);
    const uint32_t d0 = get_arg_val<uint32_t>(3) * DT;
    const uint32_t R0 = get_arg_val<uint32_t>(4);
    constexpr uint32_t TB = 2048;
    Noc noc;
    CircularBuffer out(c_OUT);
    if constexpr (T == 32) {
        for (uint32_t d = 0; d < DT; ++d) {
            out.wait_front(1);
            noc.async_write(out, sO, TB, {.offset_bytes = 0}, {.page_id = (g * GH + r) * HDT + d0 + d});
            noc.async_write_barrier();
            out.pop_front(1);
        }
    } else {
        // rows h'*T .. h'*T+T-1 of the tile lie in one face row block (16 % T == 0): two contiguous T*32-byte
        // runs (left / right face) -> rows 0..T-1 of the head's output tile. Rows >= T of OUT stay zero.
        constexpr uint32_t HPT = 32 / T;
        out.wait_front(DT);
        const uint32_t h0 = r * HPT;
        const uint32_t nh = (GH - h0 < HPT) ? GH - h0 : HPT;
        for (uint32_t hh = 0; hh < nh; ++hh) {
            const uint32_t rr = hh * T;
            for (uint32_t d = 0; d < DT; ++d) {
                for (uint32_t f = 0; f < 2; ++f) {
                    const uint32_t soff = d * TB + (((rr >> 4) * 2 + f) * 256 + (rr & 15) * 16) * 2;
                    noc.async_write(out, sO, T * 32, {.offset_bytes = soff},
                                    {.page_id = (g * GH + h0 + hh) * HDT + d0 + d,
                                     .offset_bytes = (((R0 >> 4) * 2 + f) * 256 + (R0 & 15) * 16) * 2});
                }
            }
        }
        noc.async_write_barrier();
        out.pop_front(DT);
    }
}

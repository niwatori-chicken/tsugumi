// Causal depthwise conv — writer, split output: column tiles [0, KT) -> Q, [KT, 2KT) -> K, [2KT, CTL) -> V, each
// [1,T,width] bf16 TILE (the flat q/k/v inputs of chunk_gated_delta_rule; no slices / reshapes afterwards).
// CT: 0 NT, 1 CTL, 2 RPS, 3 KT, then accessors Q, K, V.  RT: 0..2 addrs, 3 u0, 4 u1
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t NT = get_compile_time_arg_val(0);
    constexpr uint32_t CTL = get_compile_time_arg_val(1);
    constexpr uint32_t RPS = get_compile_time_arg_val(2);
    constexpr uint32_t KT = get_compile_time_arg_val(3);
    constexpr uint32_t VT = CTL - 2 * KT;
    constexpr auto aQ = TensorAccessorArgs<4>();
    constexpr auto aK = TensorAccessorArgs<aQ.next_compile_time_args_offset()>();
    constexpr auto aV = TensorAccessorArgs<aK.next_compile_time_args_offset()>();
    const auto sQ = TensorAccessor(aQ, get_arg_val<uint32_t>(0));
    const auto sK = TensorAccessor(aK, get_arg_val<uint32_t>(1));
    const auto sV = TensorAccessor(aV, get_arg_val<uint32_t>(2));
    const uint32_t u0 = get_arg_val<uint32_t>(3);
    const uint32_t u1 = get_arg_val<uint32_t>(4);
    Noc noc;
    CircularBuffer co(4);
    for (uint32_t u = u0; u < u1; ++u) {
        const uint32_t s = u / CTL, c = u % CTL;
        const uint32_t i0 = s * RPS, i1 = (i0 + RPS < NT) ? i0 + RPS : NT;
        for (uint32_t i = i0; i < i1; ++i) {
            co.wait_front(1);
            if (c < KT) {
                noc.async_write(co, sQ, 2048, {.offset_bytes = 0}, {.page_id = i * KT + c});
            } else if (c < 2 * KT) {
                noc.async_write(co, sK, 2048, {.offset_bytes = 0}, {.page_id = i * KT + c - KT});
            } else {
                noc.async_write(co, sV, 2048, {.offset_bytes = 0}, {.page_id = i * VT + c - 2 * KT});
            }
            noc.async_write_barrier();
            co.pop_front(1);
        }
    }
}

// Causal depthwise conv — writer: output tiles of each unit to Y [1,1,T,C] bf16 TILE.
// CT: 0 NT, 1 CTL, 2 RPS, then accessor Y.  RT: 0 addr, 1 u0, 2 u1
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t NT = get_compile_time_arg_val(0);
    constexpr uint32_t CTL = get_compile_time_arg_val(1);
    constexpr uint32_t RPS = get_compile_time_arg_val(2);
    constexpr auto aY = TensorAccessorArgs<3>();
    const auto sY = TensorAccessor(aY, get_arg_val<uint32_t>(0));
    const uint32_t u0 = get_arg_val<uint32_t>(1);
    const uint32_t u1 = get_arg_val<uint32_t>(2);
    Noc noc;
    CircularBuffer co(4);
    for (uint32_t u = u0; u < u1; ++u) {
        const uint32_t s = u / CTL, c = u % CTL;
        const uint32_t i0 = s * RPS, i1 = (i0 + RPS < NT) ? i0 + RPS : NT;
        for (uint32_t i = i0; i < i1; ++i) {
            co.wait_front(1);
            noc.async_write(co, sY, 2048, {.offset_bytes = 0}, {.page_id = i * CTL + c});
            noc.async_write_barrier();
            co.pop_front(1);
        }
    }
}

// Inkling decode MoE: softcapped SwiGLU — writer: row r of the dense result tile (CB 16) -> row 0 of ACT page
// o0 + r (rows 1..31 of ACT are not written). CT: accessor ACT (L1).  RT: 0 act, 1 o0, 2 n
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr auto aA = TensorAccessorArgs<0>();
    const auto sA = TensorAccessor(aA, get_arg_val<uint32_t>(0));
    const uint32_t o0 = get_arg_val<uint32_t>(1), n = get_arg_val<uint32_t>(2);
    Noc noc;
    CircularBuffer co(16);
    co.wait_front(1);
    for (uint32_t r = 0; r < n; ++r) {
        const uint32_t s0 = (r >> 4) * 1024 + (r & 15) * 32;
        for (uint32_t h = 0; h < 2; ++h)
            noc.async_write(co, sA, 32, {.offset_bytes = s0 + h * 512}, {.page_id = o0 + r, .offset_bytes = h * 512});
    }
    noc.async_write_barrier();
    co.pop_front(1);
}

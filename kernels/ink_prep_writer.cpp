// Inkling decode: per-head preparation — writer: the NT result tiles (head h in row h) to OUT pages 0..NT-1, whole
// or (a role split over cores) only the rows of heads h0 .. h0 + hn - 1.
// CT: accessor OUT (interleaved).  RT: 0 out addr, 1 nt, 2 h0, 3 hn, 4 whole tiles
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t c_O = 16;

void kernel_main() {
    constexpr auto aO = TensorAccessorArgs<0>();
    const auto sO = TensorAccessor(aO, get_arg_val<uint32_t>(0));
    const uint32_t nt = get_arg_val<uint32_t>(1), h0 = get_arg_val<uint32_t>(2), hn = get_arg_val<uint32_t>(3);
    const uint32_t full = get_arg_val<uint32_t>(4);
    Noc noc;
    CircularBuffer co(c_O);
    for (uint32_t d = 0; d < nt; ++d) {
        co.wait_front(1);
        if (full) {
            noc.async_write(co, sO, 2048, {.offset_bytes = 0}, {.page_id = d});
        } else {  // only this core's head rows (32-byte face rows; other cores write the others)
            for (uint32_t h = h0; h < h0 + hn; ++h)
                for (uint32_t f = 0; f < 2; ++f) {
                    const uint32_t o = (((h >> 4) * 2 + f) * 256 + (h & 15) * 16) * 2;
                    noc.async_write(co, sO, 32, {.offset_bytes = o}, {.page_id = d, .offset_bytes = o});
                }
        }
        noc.async_write_barrier();
        co.pop_front(1);
    }
}

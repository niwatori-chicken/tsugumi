// Inkling decode: per-head preparation — writer: the NT result tiles (head h in row h) to OUT pages 0..NT-1.
// CT: accessor OUT (interleaved).  RT: 0 out addr, 1 nt
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t c_O = 16;

void kernel_main() {
    constexpr auto aO = TensorAccessorArgs<0>();
    const auto sO = TensorAccessor(aO, get_arg_val<uint32_t>(0));
    const uint32_t nt = get_arg_val<uint32_t>(1);
    Noc noc;
    CircularBuffer co(c_O);
    for (uint32_t d = 0; d < nt; ++d) {
        co.wait_front(1);
        noc.async_write(co, sO, 2048, {.offset_bytes = 0}, {.page_id = d});
        noc.async_write_barrier();
        co.pop_front(1);
    }
}

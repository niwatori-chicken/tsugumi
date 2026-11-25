// Row gather — writer: OUT [1,1,32,C] bf16 TILE, column tiles c0..c1-1.  CT: accessor OUT.  RT: 0 addr, 1 c0, 2 c1
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr auto aO = TensorAccessorArgs<0>();
    const auto sO = TensorAccessor(aO, get_arg_val<uint32_t>(0));
    const uint32_t c0 = get_arg_val<uint32_t>(1);
    const uint32_t c1 = get_arg_val<uint32_t>(2);
    Noc noc;
    CircularBuffer co(2);
    for (uint32_t c = c0; c < c1; ++c) {
        co.wait_front(1);
        noc.async_write(co, sO, 2048, {.offset_bytes = 0}, {.page_id = c});
        noc.async_write_barrier();
        co.pop_front(1);
    }
}

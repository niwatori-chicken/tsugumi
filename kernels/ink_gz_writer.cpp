// Inkling message gate, phase A — writer: RT + 1 float32 tiles (z part, ss part) to PART pages c * (RT + 1) ..
// CT: accessor PART (DRAM).  RT: 0 part, 1 c, 2 RT
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr auto aP = TensorAccessorArgs<0>();
    const auto sP = TensorAccessor(aP, get_arg_val<uint32_t>(0));
    const uint32_t c = get_arg_val<uint32_t>(1), RT = get_arg_val<uint32_t>(2);
    Noc noc;
    CircularBuffer co(16);
    for (uint32_t i = 0; i < RT + 1; ++i) {
        co.wait_front(1);
        noc.async_write(co, sP, 4096, {.offset_bytes = 0}, {.page_id = c * (RT + 1) + i});
        noc.async_write_barrier();
        co.pop_front(1);
    }
}

// Inkling decode: short conv + residual — writer: out tiles (CB 16) to OUT, new history tiles (CB 17) to HIST.
// CT: accessors OUT (L1), HIST (DRAM).  RT: 0 out, 1 hist, 2 j0, 3 NT
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr auto aO = TensorAccessorArgs<0>();
    constexpr auto aH = TensorAccessorArgs<aO.next_compile_time_args_offset()>();
    const auto sO = TensorAccessor(aO, get_arg_val<uint32_t>(0));
    const auto sH = TensorAccessor(aH, get_arg_val<uint32_t>(1));
    const uint32_t j0 = get_arg_val<uint32_t>(2), NT = get_arg_val<uint32_t>(3);
    Noc noc;
    CircularBuffer co(16), chn(17);
    for (uint32_t j = 0; j < NT; ++j) {
        co.wait_front(1);
        noc.async_write(co, sO, 2048, {.offset_bytes = 0}, {.page_id = j0 + j});
        chn.wait_front(1);
        noc.async_write(chn, sH, 2048, {.offset_bytes = 0}, {.page_id = j0 + j});
        noc.async_write_barrier();
        co.pop_front(1);
        chn.pop_front(1);
    }
}

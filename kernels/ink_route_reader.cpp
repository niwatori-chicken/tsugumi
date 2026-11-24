// Inkling decode MoE routing — reader: router logits [1,1,1,E] float32 (L1, E/32 tiles) and the expert bias
// [1,1,1,E] float32 (DRAM) into CBs 0 / 1.  CT: accessors LOGITS, BIAS.  RT: 0 logits, 1 bias, 2 tiles
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr auto aL = TensorAccessorArgs<0>();
    constexpr auto aB = TensorAccessorArgs<aL.next_compile_time_args_offset()>();
    const auto sL = TensorAccessor(aL, get_arg_val<uint32_t>(0));
    const auto sB = TensorAccessor(aB, get_arg_val<uint32_t>(1));
    const uint32_t nt = get_arg_val<uint32_t>(2);
    Noc noc;
    CircularBuffer cl(0), cb(1);
    cl.reserve_back(nt);
    cb.reserve_back(nt);
    for (uint32_t t = 0; t < nt; ++t) {
        noc.async_read(sL, cl, 4096, {.page_id = t}, {.offset_bytes = t * 4096});
        noc.async_read(sB, cb, 4096, {.page_id = t}, {.offset_bytes = t * 4096});
    }
    noc.async_read_barrier();
    cl.push_back(nt);
    cb.push_back(nt);
}

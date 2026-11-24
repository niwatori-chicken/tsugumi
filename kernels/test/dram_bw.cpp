// DRAM read bandwidth test: core c reads pages p = c, c + NC, c + 2 NC, ... (< NP) of an interleaved tensor into a
// ring of NB page slots (CB 0), keeping up to NB reads in flight. RT: 0 addr, 1 c, 2 NC, 3 NP, 4 page bytes, 5 NB
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr auto aW = TensorAccessorArgs<0>();
    const auto sW = TensorAccessor(aW, get_arg_val<uint32_t>(0));
    const uint32_t c = get_arg_val<uint32_t>(1), NC = get_arg_val<uint32_t>(2), NP = get_arg_val<uint32_t>(3);
    const uint32_t pb = get_arg_val<uint32_t>(4), NB = get_arg_val<uint32_t>(5);
    Noc noc;
    CircularBuffer cb(0);
    cb.reserve_back(NB);
    uint32_t slot = 0, n = 0;
    for (uint32_t p = c; p < NP; p += NC) {
        noc.async_read(sW, cb, pb, {.page_id = p}, {.offset_bytes = slot * pb});
        if (++slot == NB) slot = 0;
        if (++n % NB == 0) noc.async_read_barrier();
    }
    noc.async_read_barrier();
}

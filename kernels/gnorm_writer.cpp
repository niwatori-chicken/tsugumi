// Gated RMSNorm on the head-major GDN output — writer: OUT [1,1,T,NV*128] bf16 TILE, tiles tt*NV*4 + h*4 + j.
// CT: 0 NV, then accessor OUT.  RT: 0 addr, 1 u0, 2 u1
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t NV = get_compile_time_arg_val(0);
    constexpr auto aOut = TensorAccessorArgs<1>();
    const auto sOut = TensorAccessor(aOut, get_arg_val<uint32_t>(0));
    const uint32_t u0 = get_arg_val<uint32_t>(1);
    const uint32_t u1 = get_arg_val<uint32_t>(2);
    Noc noc;
    CircularBuffer co(5);
    for (uint32_t u = u0; u < u1; ++u) {
        const uint32_t tt = u / NV, h = u % NV;
        co.wait_front(4);
        for (uint32_t j = 0; j < 4; ++j) {
            noc.async_write(co, sOut, 2048, {.offset_bytes = j * 2048}, {.page_id = tt * NV * 4 + h * 4 + j});
        }
        noc.async_write_barrier();
        co.pop_front(4);
    }
}

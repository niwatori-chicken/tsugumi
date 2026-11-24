// Inkling message gate, phase B — writer: CPT bf16 tiles to OUT pages j0 ..  CT: accessor OUT.  RT: 0 out, 1 j0, 2 CPT
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr auto aO = TensorAccessorArgs<0>();
    const auto sO = TensorAccessor(aO, get_arg_val<uint32_t>(0));
    const uint32_t j0 = get_arg_val<uint32_t>(1), CPT = get_arg_val<uint32_t>(2);
    Noc noc;
    CircularBuffer co(16);
    for (uint32_t j = 0; j < CPT; ++j) {
        co.wait_front(1);
        noc.async_write(co, sO, 2048, {.offset_bytes = 0}, {.page_id = j0 + j});
        noc.async_write_barrier();
        co.pop_front(1);
    }
}

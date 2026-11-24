// Inkling message gate in one program — writer: arrive at core 0 once this core's partials (CB 4) exist; the CPT
// output tiles -> OUT; a final arrival + wait keeps CB 4 alive until every core has read it.
// CT: accessor OUT.  RT: 0 out, 1 j0, 2 CPT, 3 RT, 4 core 0 x, 5 core 0 y
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr auto aO = TensorAccessorArgs<0>();
    const auto sO = TensorAccessor(aO, get_arg_val<uint32_t>(0));
    const uint32_t j0 = get_arg_val<uint32_t>(1), CPT = get_arg_val<uint32_t>(2), RT = get_arg_val<uint32_t>(3);
    const uint64_t arrive = get_noc_addr(get_arg_val<uint32_t>(4), get_arg_val<uint32_t>(5), get_semaphore(0));
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    Noc noc;
    CircularBuffer cpa(4), co(16);
    cpa.wait_front(RT + 1);
    noc_semaphore_inc(arrive, 1);
    for (uint32_t j = 0; j < CPT; ++j) {
        co.wait_front(1);
        noc.async_write(co, sO, 2048, {.offset_bytes = 0}, {.page_id = j0 + j});
        noc.async_write_barrier();
        co.pop_front(1);
    }
    noc_semaphore_inc(arrive, 1);
    noc_semaphore_wait_min(go, 2);
    cpa.pop_front(RT + 1);
    noc_async_atomic_barrier();
}

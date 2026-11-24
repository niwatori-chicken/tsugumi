// Inkling prefill MoE — writer: per batch, once the compute has made its NB shared act tiles (CB 13), arrive at
// core 0's ARRIVE semaphore; after the batch's release (GO >= b + 1) the previous batch's share slots are free
// (every core has finished reading them). Then the 2 output tiles (CB 16) -> OUT pages 2c, 2c + 1.
// A final round (arrive once more, wait for GO = batches + 1) keeps this core's share slots alive until every core
// has read them. The number of used experts comes from the reader (CB 11).
// CT: accessor OUT.  RT: 0 out, 1 c, 2 coord x, 3 coord y (core 0's NoC coordinates), 4 NB
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr auto aO = TensorAccessorArgs<0>();
    const auto sO = TensorAccessor(aO, get_arg_val<uint32_t>(0));
    const uint32_t c = get_arg_val<uint32_t>(1), cx = get_arg_val<uint32_t>(2), cy = get_arg_val<uint32_t>(3);
    const uint32_t NB = get_arg_val<uint32_t>(4);
    Noc noc;
    CircularBuffer csh(13), co(16), cu(11);
    cu.wait_front(1);
    const uint32_t U = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cu.get_read_ptr())[0];
    cu.pop_front(1);
    const uint32_t nbatch = (U + NB - 1) / NB;
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    const uint64_t arrive = get_noc_addr(cx, cy, get_semaphore(0));
    uint32_t prev = 0;  // tiles of the previous batch still in SHARE
    for (uint32_t b = 0; b < nbatch; ++b) {
        const uint32_t n = (b * NB + NB < U) ? NB : U - b * NB;
        csh.wait_front(prev + n);
        noc_semaphore_inc(arrive, 1);
        noc_semaphore_wait_min(go, b + 1);
        if (prev) csh.pop_front(prev);
        prev = n;
    }
    co.wait_front(2);
    for (uint32_t j = 0; j < 2; ++j) noc.async_write(co, sO, 2048, {.offset_bytes = j * 2048}, {.page_id = 2 * c + j});
    noc.async_write_barrier();
    co.pop_front(2);
    // the last batch's tiles stay until every core has read them: wait for all cores to finish (one more round)
    noc_semaphore_inc(arrive, 1);
    noc_semaphore_wait_min(go, nbatch + 1);
    if (prev) csh.pop_front(prev);
    noc_async_atomic_barrier();
}

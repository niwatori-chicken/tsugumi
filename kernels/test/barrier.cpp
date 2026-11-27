// Global barrier cost test: every core increments the coordinator's ARRIVE semaphore, the coordinator waits for
// all of them and multicasts the round number into every core's GO semaphore (two rectangles: the worker columns
// are split in NoC x). RT: 0 coord x, 1 coord y, 2 is_coord, 3 ncores, 4 rounds, 5..8 rect A (x0 y0 x1 y1),
// 9 dests A, 10..13 rect B, 14 dests B (0 = no rect B)
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t cx = get_arg_val<uint32_t>(0), cy = get_arg_val<uint32_t>(1), is_coord = get_arg_val<uint32_t>(2);
    const uint32_t ncores = get_arg_val<uint32_t>(3), rounds = get_arg_val<uint32_t>(4);
    const uint32_t arrive = get_semaphore(0), go = get_semaphore(1);
    volatile tt_l1_ptr uint32_t* parrive = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrive);
    volatile tt_l1_ptr uint32_t* pgo = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go);
    const uint64_t coord_arrive = get_noc_addr(cx, cy, arrive);
    for (uint32_t r = 1; r <= rounds; ++r) {
        noc_semaphore_inc(coord_arrive, 1);
        if (is_coord) {
            noc_semaphore_wait_min(parrive, ncores * r);
            *pgo = r;
            const uint64_t a = get_noc_multicast_addr(get_arg_val<uint32_t>(5), get_arg_val<uint32_t>(6),
                                                      get_arg_val<uint32_t>(7), get_arg_val<uint32_t>(8), go);
            noc_semaphore_set_multicast(go, a, get_arg_val<uint32_t>(9));
            if (get_arg_val<uint32_t>(14)) {
                const uint64_t b = get_noc_multicast_addr(get_arg_val<uint32_t>(10), get_arg_val<uint32_t>(11),
                                                          get_arg_val<uint32_t>(12), get_arg_val<uint32_t>(13), go);
                noc_semaphore_set_multicast(go, b, get_arg_val<uint32_t>(14));
            }
            noc_async_write_barrier();
        }
        noc_semaphore_wait_min(pgo, r);
    }
    noc_async_atomic_barrier();
}

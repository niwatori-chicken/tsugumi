// Inkling decode MoE router + routing in one program (12 cores) — reader.
// Core c: logits column tile n = c / KS of the K third ks = c % KS: x tiles (row 0) and router weight tiles
// (Wg [H, E] bf16, page kt * NT + n). Core 0 additionally (after every core's partial exists) pulls the 12
// partial logits tiles (float32) from the cores' CB 4 and reads the expert bias, then releases the final round.
// CB 0 X (KTC), 1 W (KTC), 5 P (NT * KS partials, core 0), 6 BIAS (NT, core 0), 4 PA (base address only).
// Semaphores 0 ARRIVE (core 0), 1 GO.  CT: accessors X, W, BIAS.
// RT: 0 x, 1 w, 2 bias, 3 c, 4 kt0, 5 KTC, 6 n, 7 NT, 8 NC, 9.. NoC (x, y) of every core
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr auto aX = TensorAccessorArgs<0>();
    constexpr auto aW = TensorAccessorArgs<aX.next_compile_time_args_offset()>();
    constexpr auto aB = TensorAccessorArgs<aW.next_compile_time_args_offset()>();
    const auto sX = TensorAccessor(aX, get_arg_val<uint32_t>(0));
    const auto sW = TensorAccessor(aW, get_arg_val<uint32_t>(1));
    const auto sB = TensorAccessor(aB, get_arg_val<uint32_t>(2));
    const uint32_t c = get_arg_val<uint32_t>(3), kt0 = get_arg_val<uint32_t>(4), KTC = get_arg_val<uint32_t>(5);
    const uint32_t n = get_arg_val<uint32_t>(6), NT = get_arg_val<uint32_t>(7), NC = get_arg_val<uint32_t>(8);
    constexpr uint32_t XY = 9, TB = 2048;
    Noc noc;
    CircularBuffer cx(0), cw(1), cpa(4), cp(5), cb(6);
    const uint32_t pa = cpa.get_read_ptr();
    cx.reserve_back(KTC);
    cw.reserve_back(KTC);
    for (uint32_t j = 0; j < KTC; ++j) {
        noc.async_read(sX, cx, TB, {.page_id = kt0 + j}, {.offset_bytes = j * TB});
        noc.async_read(sW, cw, TB, {.page_id = (kt0 + j) * NT + n}, {.offset_bytes = j * TB});
    }
    noc.async_read_barrier();
    cx.push_back(KTC);
    cw.push_back(KTC);
    if (c != 0) return;
    cb.reserve_back(NT);
    for (uint32_t t = 0; t < NT; ++t) noc.async_read(sB, cb, 4096, {.page_id = t}, {.offset_bytes = t * 4096});
    noc.async_read_barrier();
    cb.push_back(NT);
    volatile tt_l1_ptr uint32_t* arrive = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    noc_semaphore_wait_min(arrive, NC);
    cp.reserve_back(NC);
    const uint32_t pl = cp.get_write_ptr();
    for (uint32_t r = 0; r < NC; ++r)  // partial of core r = (n = r / KS, ks = r % KS) at slot r
        noc_async_read(get_noc_addr(get_arg_val<uint32_t>(XY + 2 * r), get_arg_val<uint32_t>(XY + 2 * r + 1), pa),
                       pl + r * 4096, 4096);
    noc_async_read_barrier();
    cp.push_back(NC);
    // final round: every partial has been read
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    *go = 1;
    for (uint32_t i = 1; i < NC; ++i)
        noc_semaphore_set_remote(get_semaphore(1), get_noc_addr(get_arg_val<uint32_t>(XY + 2 * i),
                                                                get_arg_val<uint32_t>(XY + 2 * i + 1), get_semaphore(1)));
    noc_async_write_barrier();
}

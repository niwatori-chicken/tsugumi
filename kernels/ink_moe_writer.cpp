// Inkling decode MoE — writer: dense gate / up tiles (row k = row 0 of expert k's tile), the act tile to the
// gather buffer (page c) + the arrival at core 0's ARRIVE semaphore (its reader releases GO), the 2 output tiles.
// Semaphores: 0 ARRIVE (core 0), 1 GO.  CT: accessors GATHER (L1), OUT (L1).
// RT: 0 gather, 1 out, 2 c, 3 coord x, 4 coord y, 5 ncores, 6..9 multicast rect (x0 y0 x1 y1), 10 dests,
//     11 stage (debug: 3 = all)
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t K = 8;

FORCE_INLINE void copy32(uint32_t dst, uint32_t src) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    for (uint32_t i = 0; i < 8; ++i) d[i] = s[i];
}

void kernel_main() {
    constexpr auto aG = TensorAccessorArgs<0>();
    constexpr auto aO = TensorAccessorArgs<aG.next_compile_time_args_offset()>();
    const auto sG = TensorAccessor(aG, get_arg_val<uint32_t>(0));
    const auto sO = TensorAccessor(aO, get_arg_val<uint32_t>(1));
    const uint32_t c = get_arg_val<uint32_t>(2), cx = get_arg_val<uint32_t>(3), cy = get_arg_val<uint32_t>(4);
    const uint32_t ncores = get_arg_val<uint32_t>(5);
    Noc noc;
    CircularBuffer cg(4), cu(5), cdg(8), cdu(9), ca(10), co(16);
    cdg.reserve_back(1);
    cdu.reserve_back(1);
    const uint32_t dg = cdg.get_write_ptr(), du = cdu.get_write_ptr();
    {
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dg);
        volatile tt_l1_ptr uint32_t* z2 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(du);
        for (uint32_t i = 0; i < 512; ++i) z[i] = z2[i] = 0;
    }
    for (uint32_t k = 0; k < K; ++k) {
        cg.wait_front(1);
        cu.wait_front(1);
        const uint32_t g = cg.get_read_ptr(), u = cu.get_read_ptr();
        for (uint32_t f = 0; f < 2; ++f) {  // row 0 of faces 0 / 1 -> row k
            copy32(dg + f * 512 + k * 32, g + f * 512);
            copy32(du + f * 512 + k * 32, u + f * 512);
        }
        cg.pop_front(1);
        cu.pop_front(1);
    }
    const uint32_t stage = get_arg_val<uint32_t>(11);
    if (stage == 1) return;
    cdg.push_back(1);
    cdu.push_back(1);
    ca.wait_front(1);
    noc.async_write(ca, sG, 2048, {.offset_bytes = 0}, {.page_id = c});
    noc.async_write_barrier();
    ca.pop_front(1);
    const uint32_t arrive = get_semaphore(0), go = get_semaphore(1);
    noc_semaphore_inc(get_noc_addr(cx, cy, arrive), 1);  // core 0's reader releases (NOC 0 multicast)
    (void)go;
    (void)ncores;
    if (stage == 2) {
        noc_async_atomic_barrier();
        return;
    }
    co.wait_front(2);
    for (uint32_t j = 0; j < 2; ++j) noc.async_write(co, sO, 2048, {.offset_bytes = j * 2048}, {.page_id = 2 * c + j});
    noc.async_write_barrier();
    co.pop_front(2);
    noc_async_atomic_barrier();
}

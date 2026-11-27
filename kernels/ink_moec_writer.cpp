// Inkling prefill MoE with compact token tiles — writer. Per used expert (in the compute's order): the routed
// tokens' rows of G / U (CB 4 / 5, MT tiles) copied into CT compact tiles (CB 8 / 9: row r = token L[r]; rows
// beyond n stay finite: zeroed once, then stale), and the MT x CT placement tiles PS (CB 3: [t % 32][r % 32] = 1
// for token t = L[r] in row tile t / 32, compact tile r / 32). Per batch, once the compute has made its NB x MT
// shared act tiles (CB 13): arrive at core 0, wait for the release, free the previous batch's slots. Then the 2 MT
// output tiles (CB 16) -> OUT pages m * 72 + 2c + j; a final round keeps CB 13 alive until every core has read it.
// CB 11: U (from the reader), 17: token lists (reader: per used expert j, n_j then the token ids, T + 1 bytes).
// CT: accessor OUT.  RT: 0 out, 1 c, 2 core 0 x, 3 core 0 y, 4 NB, 5 MT
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

FORCE_INLINE uint32_t elem_off(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

FORCE_INLINE void copy_row(uint32_t dst_tile, uint32_t dr, uint32_t src_tile, uint32_t sr) {  // bf16 tile rows
    for (uint32_t f = 0; f < 2; ++f) {
        volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            dst_tile + (((dr >> 4) * 2 + f) * 256 + (dr & 15) * 16) * 2);
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            src_tile + (((sr >> 4) * 2 + f) * 256 + (sr & 15) * 16) * 2);
        for (uint32_t i = 0; i < 8; ++i) d[i] = s[i];
    }
}

void kernel_main() {
    constexpr auto aO = TensorAccessorArgs<0>();
    const auto sO = TensorAccessor(aO, get_arg_val<uint32_t>(0));
    const uint32_t c = get_arg_val<uint32_t>(1), cx = get_arg_val<uint32_t>(2), cy = get_arg_val<uint32_t>(3);
    const uint32_t NB = get_arg_val<uint32_t>(4), MT = get_arg_val<uint32_t>(5);
    constexpr uint32_t TB = 2048;
    const uint32_t CTM = MT < 2 ? MT : 2, LS = 32 * CTM + 1;  // compact tiles per entry, list stride
    Noc noc;
    CircularBuffer cps(3), cg(4), cu(5), ccg(8), ccu(9), csh(13), co(16), cuu(11), cls(17);
    {  // compact tiles start finite (rows beyond n are never written)
        for (CircularBuffer* cb : {&ccg, &ccu}) {
            cb->reserve_back(2 * CTM);
            volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb->get_write_ptr());
            for (uint32_t i = 0; i < 2 * CTM * 512; ++i) z[i] = 0;
        }
    }
    cuu.wait_front(1);
    const uint32_t U = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cuu.get_read_ptr())[0];
    cuu.pop_front(1);
    cls.wait_front(1);
    volatile tt_l1_ptr uint8_t* lists = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(cls.get_read_ptr());
    const uint32_t nbatch = (U + NB - 1) / NB;
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    const uint64_t arrive = get_noc_addr(cx, cy, get_semaphore(0));
    uint32_t prev = 0;  // tiles of the previous batch still in SHARE
    for (uint32_t b = 0; b < nbatch; ++b) {
        const uint32_t j0 = b * NB, n = (j0 + NB < U) ? NB : U - j0;
        for (uint32_t j = j0; j < j0 + n; ++j) {
            const uint32_t cnt = lists[j * LS], ct = (cnt + 31) / 32;
            volatile tt_l1_ptr uint8_t* L = lists + j * LS + 1;
            cg.wait_front(MT);
            cu.wait_front(MT);
            ccg.reserve_back(CTM);  // fixed CTM-tile blocks (ct used)
            ccu.reserve_back(CTM);
            const uint32_t g = cg.get_read_ptr(), u = cu.get_read_ptr();
            const uint32_t dg = ccg.get_write_ptr(), du = ccu.get_write_ptr();
            for (uint32_t r = 0; r < cnt; ++r) {
                const uint32_t t = L[r];
                copy_row(dg + (r >> 5) * TB, r & 31, g + (t >> 5) * TB, t & 31);
                copy_row(du + (r >> 5) * TB, r & 31, u + (t >> 5) * TB, t & 31);
            }
            ccg.push_back(CTM);
            ccu.push_back(CTM);
            cg.pop_front(MT);
            cu.pop_front(MT);
            // placement tiles (m, mi) at m * CTM + mi (fixed MT x CTM blocks; mi < ct used)
            cps.reserve_back(MT * CTM);
            const uint32_t ps = cps.get_write_ptr();
            volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ps);
            for (uint32_t m = 0; m < MT; ++m)
                for (uint32_t i = 0; i < ct * 512; ++i) z[(m * CTM) * 512 + i] = 0;
            volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ps);
            for (uint32_t r = 0; r < cnt; ++r) {
                const uint32_t t = L[r];
                p[((t >> 5) * CTM + (r >> 5)) * 1024 + elem_off(t & 31, r & 31)] = 0x3F80;  // 1.0
            }
            cps.push_back(MT * CTM);
        }
        csh.wait_front(prev + n * CTM);
        noc_semaphore_inc(arrive, 1);
        noc_semaphore_wait_min(go, b + 1);
        if (prev) csh.pop_front(prev);
        prev = n * CTM;
    }
    co.wait_front(2 * MT);
    for (uint32_t m = 0; m < MT; ++m)
        for (uint32_t jo = 0; jo < 2; ++jo)
            noc.async_write(co, sO, 2048, {.offset_bytes = (2 * m + jo) * 2048}, {.page_id = m * 72 + 2 * c + jo});
    noc.async_write_barrier();
    co.pop_front(2 * MT);
    noc_semaphore_inc(arrive, 1);
    noc_semaphore_wait_min(go, nbatch + 1);
    if (prev) csh.pop_front(prev);
    cls.pop_front(1);
    noc_async_atomic_barrier();
}

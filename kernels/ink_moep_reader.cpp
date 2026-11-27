// Inkling prefill MoE (MT x 32 token rows) on the bank-contiguous expert layout (ink_moe_reorder.cpp) — reader.
// Same cores and weight streams as ink_moe_reader.cpp (core c: act column tile c, output tiles 2c, 2c + 1), but
// for every USED expert (any token routed to it), in batches of NB experts:
//   gate_up weights of the batch; then, after the batch barrier, per expert the 36 act tiles (32 token rows,
//   already scaled by the routing weights) read straight from the producing cores' share CB, and the down weights.
// CB 0 X (72), 1 W (weight ring), 2 D (72: two experts' act tiles), 6 ROUTE (4 tiles: dense routing weights
//    [32 tokens, 128 experts]), 7 scratch, 11 / 12 number of used experts for the writer / compute, 14 WC (per expert:
//    column 0 = the 32 tokens' routing weights), 13 SHARE (not touched here: its base address on every core).
// Semaphores: 0 ARRIVE (core 0), 1 GO (number of batches released).
// CT: 0 B, 1 NBUF, 2 MAXF, 3 MT, accessors X, ROUTE, USED (row-major bf16 [1,1,1,128]: max routing weight per expert).
// RT: 0 x, 1 route, 2 used, 3 gu base, 4 dn base, 5 c, 6 page bytes, 7 NB, 8 ncores, 9.. NoC (x, y) of each core
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t B = get_compile_time_arg_val(0), NBUF = get_compile_time_arg_val(1), MAXF = get_compile_time_arg_val(2);
constexpr uint32_t MT = get_compile_time_arg_val(3);  // row tiles (32 tokens each)
constexpr uint32_t E = 128, KT = 72, IT = 36;

FORCE_INLINE uint32_t elem_off(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

uint32_t g_dbg = 0;  // debug bits: 1 no weight reads, 4 no act-tile gather

struct Ring {
    CircularBuffer& cw;
    uint32_t pb, base, wr = 0, inflight = 0, issued = 0;
    Ring(CircularBuffer& c, uint32_t pb_) : cw(c), pb(pb_), base(c.get_write_ptr()) {}
    void push_oldest() {
        noc_async_read_barrier_with_trid(1 + (issued - inflight) % NBUF);
        cw.push_back(B);
        wr = (wr + 1) % NBUF;
        --inflight;
    }
    void issue(uint32_t bank, uint32_t addr) {
        if (inflight == MAXF) push_oldest();
        cw.reserve_back((inflight + 1) * B);
        const uint32_t l1 = base + ((wr + inflight) % NBUF) * B * pb;
        if (!(g_dbg & 1)) {
            noc_async_read_set_trid(1 + issued % NBUF);
            noc_async_read(get_noc_addr_from_bank_id<true>(bank, addr), l1, B * pb);
        }
        ++inflight;
        ++issued;
    }
    void flush() {
        while (inflight) push_oldest();
    }
};

void kernel_main() {
    constexpr auto aX = TensorAccessorArgs<4>();
    constexpr auto aR = TensorAccessorArgs<aX.next_compile_time_args_offset()>();
    constexpr auto aU = TensorAccessorArgs<aR.next_compile_time_args_offset()>();
    const auto sX = TensorAccessor(aX, get_arg_val<uint32_t>(0));
    const auto sR = TensorAccessor(aR, get_arg_val<uint32_t>(1));
    const auto sU = TensorAccessor(aU, get_arg_val<uint32_t>(2));
    const uint32_t gu = get_arg_val<uint32_t>(3), dn = get_arg_val<uint32_t>(4), c = get_arg_val<uint32_t>(5);
    const uint32_t pb = get_arg_val<uint32_t>(6), NB = get_arg_val<uint32_t>(7), ncores = get_arg_val<uint32_t>(8);
    g_dbg = get_arg_val<uint32_t>(9 + 2 * ncores);
    constexpr uint32_t TB = 2048;
    Noc noc;
    CircularBuffer cx(0), cw(1), cd(2), cr(6), csc(7), cctl(12), csh(13), cwc(14);
    const uint32_t share = csh.get_read_ptr();
    csc.reserve_back(1);
    const uint32_t sb = csc.get_write_ptr();
    noc.async_read(sU, csc, E * 2, {.page_id = 0}, {.offset_bytes = 0});
    cr.reserve_back(4 * MT);
    for (uint32_t j = 0; j < 4 * MT; ++j) noc.async_read(sR, cr, TB, {.page_id = j}, {.offset_bytes = j * TB});
    cx.reserve_back(KT * MT);  // tile (m, kt) at m * KT + kt (row-major tiles of [T, H])
    for (uint32_t j = 0; j < KT * MT; ++j) noc.async_read(sX, cx, TB, {.page_id = j}, {.offset_bytes = j * TB});
    noc.async_read_barrier();
    cx.push_back(KT * MT);
    // used experts
    volatile tt_l1_ptr uint16_t* used = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sb);
    uint32_t ue[E], U = 0;
    for (uint32_t e = 0; e < E; ++e)
        if (used[e] & 0x7FFF) ue[U++] = e;
    CircularBuffer cctw(11);
    cctl.reserve_back(1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cctl.get_write_ptr())[0] = U;
    cctl.push_back(1);
    cctw.reserve_back(1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cctw.get_write_ptr())[0] = U;
    cctw.push_back(1);
    volatile tt_l1_ptr uint16_t* route = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cr.get_write_ptr());
    const uint32_t first = (c / 4) % 2;
    const uint32_t slot = c / 4;
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    Ring ring(cw, pb);
    const uint32_t nbatch = (U + NB - 1) / NB;
    for (uint32_t b = 0; b < nbatch; ++b) {
        const uint32_t j0 = b * NB, j1 = (j0 + NB < U) ? j0 + NB : U;
        for (uint32_t j = j0; j < j1; ++j) {
            const uint32_t e = ue[j];
            cwc.reserve_back(MT);  // per row tile: column 0 = routing weights of its 32 tokens (other columns unused)
            volatile tt_l1_ptr uint16_t* wc = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cwc.get_write_ptr());
            for (uint32_t m = 0; m < MT; ++m)
                for (uint32_t t = 0; t < 32; ++t)
                    wc[m * 1024 + elem_off(t, 0)] = route[(m * 4 + (e >> 5)) * 1024 + elem_off(t, e & 31)];
            cwc.push_back(MT);
            for (uint32_t hh = 0; hh < 2; ++hh) {
                const uint32_t h = hh ^ first;
                const uint32_t base = gu + (e * 9 + slot) * 72 * pb;
                for (uint32_t g = 0; g < 72 / B; ++g) ring.issue(c % 4 + 4 * h, base + g * B * pb);
            }
        }
        ring.flush();  // the compute needs the whole batch's gate_up before the barrier
        const uint32_t base0 = dn + (ue[j0] * 9 + slot) * 36 * pb;
        for (uint32_t g = 0; g < MAXF - 1 && g < 36 / B; ++g) ring.issue(c % 4 + 4 * first, base0 + g * B * pb);
        if (c == 0) {  // coordinator: release batch b once every core's writer arrived
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), ncores * (b + 1));
            *go = b + 1;
            for (uint32_t i = 1; i < ncores; ++i)
                noc_semaphore_set_remote(get_semaphore(1), get_noc_addr(get_arg_val<uint32_t>(9 + 2 * i),
                                                                        get_arg_val<uint32_t>(10 + 2 * i), get_semaphore(1)));
            noc_async_write_barrier();
        }
        noc_semaphore_wait_min(go, b + 1);
        for (uint32_t j = j0; j < j1; ++j) {
            const uint32_t e = ue[j];
            const uint32_t sa = share + ((b % 2) * NB + (j - j0)) * MT * TB;  // this expert's act tiles, every core
            cd.reserve_back(IT * MT);  // tile (m, kt) at m * IT + kt
            const uint32_t dl = cd.get_write_ptr();
            for (uint32_t kt = 0; kt < IT && !(g_dbg & 4); ++kt)
                for (uint32_t m = 0; m < MT; ++m)
                    noc_async_read(get_noc_addr(get_arg_val<uint32_t>(9 + 2 * kt), get_arg_val<uint32_t>(10 + 2 * kt),
                                                sa + m * TB), dl + (m * IT + kt) * TB, TB);
            noc_async_read_barrier();
            cd.push_back(IT * MT);
            for (uint32_t hh = 0; hh < 2; ++hh) {
                const uint32_t h = hh ^ first;
                const uint32_t base = dn + (e * 9 + slot) * 36 * pb;
                for (uint32_t g = 0; g < 36 / B; ++g) {
                    if (j == j0 && hh == 0 && g < MAXF - 1) continue;  // prefetched
                    ring.issue(c % 4 + 4 * h, base + g * B * pb);
                }
            }
        }
    }
    ring.flush();
    if (c == 0) {  // final round: every core has read every share slot (no core leaves while others still read it)
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), ncores * (nbatch + 1));
        *go = nbatch + 1;
        for (uint32_t i = 1; i < ncores; ++i)
            noc_semaphore_set_remote(get_semaphore(1), get_noc_addr(get_arg_val<uint32_t>(9 + 2 * i),
                                                                    get_arg_val<uint32_t>(10 + 2 * i), get_semaphore(1)));
        noc_async_write_barrier();
    }
    csc.push_back(1);
    cr.push_back(4);
}

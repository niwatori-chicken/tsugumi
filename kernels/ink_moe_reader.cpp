// Inkling decode MoE (8 routed experts of one token) in one program on NC = 36 cores — reader.
// Core c: act column tile c (gate tile c, up tile c + 36 of gate_up) and output column tiles 2c, 2c + 1 (down).
// Weights in the bank-contiguous layout of ink_moe_reorder.cpp; half h of core c is in bank c % 4 + 4 h, slot
// c / 4 (banks b and b + 4 are both near the same worker rows; the placement puts each core near its banks). Streams for each selected expert: gate_up 2 halves x 6 blocks of 12 tiles (6 K tiles x [g, u]),
// then down 2 halves x 3 blocks of 12 tiles (6 K tiles x [n0, n1]) after the act barrier.
// CB 0 X (72 tiles, row 0 = x), 1 W (weights: ring of NBUF blocks of B tiles), 2 D (36 dense act tiles, rows = experts),
//    3 SEL (8 tiles: [0][e] = routing weight e), 7 scratch.
// CT: 0 B, 1 NBUF, 2 MAXF (weight ring), then accessors X (L1), IDS (L1, uint16 [1,1,1,8]), WCOL (L1, [1,8,1,1]
//     bf16 tiles), GATHER (L1 [32, 1152]).
// RT: 0 x, 1 ids, 2 wcol, 3 gu base, 4 dn base, 5 c, 6 gather, 7 page bytes (bfp8), 8 stage (debug: 3 = all),
//     9 ncores, 10.. NoC (x, y) of every core
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t K = 8, KT = 72, IT = 36;

uint32_t g_stage = 3;

// Weight blocks of B tiles into a ring of NBUF blocks (CB 1 = NBUF * B pages), up to MAXF blocks in flight, each
// read under its own NoC transaction id (as ttnn's DRAM-sharded matmul reader): reads of the next blocks overlap
// the wait for the oldest one.
constexpr uint32_t B = get_compile_time_arg_val(0), NBUF = get_compile_time_arg_val(1), MAXF = get_compile_time_arg_val(2);

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
    // B tiles from DRAM bank BANK at ADDR
    void issue(uint32_t bank, uint32_t addr) {
        if (inflight == MAXF) push_oldest();
        cw.reserve_back((inflight + 1) * B);
        const uint32_t l1 = base + ((wr + inflight) % NBUF) * B * pb;
        if (g_stage != 5) {  // (debug stage 5: no DRAM reads)
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
    constexpr auto aX = TensorAccessorArgs<3>();
    constexpr auto aI = TensorAccessorArgs<aX.next_compile_time_args_offset()>();
    constexpr auto aW = TensorAccessorArgs<aI.next_compile_time_args_offset()>();
    constexpr auto aG = TensorAccessorArgs<aW.next_compile_time_args_offset()>();
    const auto sX = TensorAccessor(aX, get_arg_val<uint32_t>(0));
    const auto sI = TensorAccessor(aI, get_arg_val<uint32_t>(1));
    const auto sW = TensorAccessor(aW, get_arg_val<uint32_t>(2));
    const auto sG = TensorAccessor(aG, get_arg_val<uint32_t>(6));
    const uint32_t gu = get_arg_val<uint32_t>(3), dn = get_arg_val<uint32_t>(4), c = get_arg_val<uint32_t>(5);
    const uint32_t pb = get_arg_val<uint32_t>(7);
    g_stage = get_arg_val<uint32_t>(8);
    constexpr uint32_t TB = 2048;
    Noc noc;
    CircularBuffer cx(0), cw(1), cd(2), cs(3), csc(7);
    // expert ids and routing weights (row 0 of each WCOL tile)
    csc.reserve_back(1);
    const uint32_t sb = csc.get_write_ptr();
    noc.async_read(sI, csc, 16, {.page_id = 0}, {.offset_bytes = 0});
    for (uint32_t k = 0; k < K; ++k) noc.async_read(sW, csc, 32, {.page_id = k}, {.offset_bytes = 64 + k * 32});
    cx.reserve_back(KT);
    for (uint32_t j = 0; j < KT; ++j) noc.async_read(sX, cx, TB, {.page_id = j}, {.offset_bytes = j * TB});
    noc.async_read_barrier();
    cx.push_back(KT);
    volatile tt_l1_ptr uint16_t* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sb);
    uint32_t e_[K];
    for (uint32_t k = 0; k < K; ++k) e_[k] = ids[k];
    // gate_up weights (72 tiles per half: 6 blocks)
    Ring ring(cw, pb);
    const uint32_t first = (c / 4) % 2;  // half read first: about half of the cores on each bank of a pair at a time
    for (uint32_t k = 0; k < K; ++k) {
        for (uint32_t hh = 0; hh < 2; ++hh) {
            const uint32_t h = hh ^ first;
            const uint32_t bank = c % 4 + 4 * h, slot = c / 4;
            const uint32_t base = gu + (e_[k] * 9 + slot) * 72 * pb;
            for (uint32_t g = 0; g < 72 / B; ++g) ring.issue(bank, base + g * B * pb);
        }
    }
    ring.flush();  // the compute needs all of gate_up before the act barrier
    const uint32_t stage = g_stage;
    if (stage == 1) return;
    // selectors: SEL_e [0][e] = w_e
    cs.reserve_back(K);
    {
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cs.get_write_ptr());
        for (uint32_t i = 0; i < K * 512; ++i) z[i] = 0;
        volatile tt_l1_ptr uint16_t* s = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cs.get_write_ptr());
        volatile tt_l1_ptr uint16_t* w = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sb + 64);
        for (uint32_t k = 0; k < K; ++k) s[k * 1024 + k] = w[k * 16];
    }
    cs.push_back(K);
    // down weights of the first expert, prefetched while the act tiles are made (MAXF - 1 blocks: none pushed, so
    // the ring never waits on the compute, which is waiting for the act tiles)
    const uint32_t b0 = c % 4 + 4 * first;
    const uint32_t base0 = dn + (e_[0] * 9 + c / 4) * 36 * pb;
    for (uint32_t g = 0; g < MAXF - 1 && g < 36 / B; ++g) ring.issue(b0, base0 + g * B * pb);
    // wait for every core's act tile, then gather the 36 dense act tiles
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    if (c == 0) {  // coordinator: all writers arrived -> GO to every core
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), get_arg_val<uint32_t>(9));
        *go = 1;
        for (uint32_t i = 1; i < get_arg_val<uint32_t>(9); ++i)  // unicast: any core placement
            noc_semaphore_set_remote(get_semaphore(1), get_noc_addr(get_arg_val<uint32_t>(10 + 2 * i),
                                                                    get_arg_val<uint32_t>(11 + 2 * i), get_semaphore(1)));
        noc_async_write_barrier();
    }
    noc_semaphore_wait(go, 1);
    if (stage == 2) return;
    cd.reserve_back(IT);
    for (uint32_t j = 0; j < IT; ++j) noc.async_read(sG, cd, TB, {.page_id = j}, {.offset_bytes = j * TB});
    noc.async_read_barrier();
    cd.push_back(IT);
    // down weights (36 tiles per half: 3 blocks; the first MAXF - 1 blocks of expert 0 are in flight already)
    for (uint32_t k = 0; k < K; ++k) {
        for (uint32_t hh = 0; hh < 2; ++hh) {
            const uint32_t h = hh ^ first;
            const uint32_t bank = c % 4 + 4 * h, slot = c / 4;
            const uint32_t base = dn + (e_[k] * 9 + slot) * 36 * pb;
            for (uint32_t g = 0; g < 36 / B; ++g) {
                if (k == 0 && hh == 0 && g < MAXF - 1) continue;
                ring.issue(bank, base + g * B * pb);
            }
        }
    }
    ring.flush();
}

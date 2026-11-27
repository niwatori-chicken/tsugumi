// Inkling prefill MoE for long chunks (T = 32 MT rows, up to 1024) — reader. Only the routed (token, expert) pairs
// are computed: per used expert its tokens are packed into compact row tiles (entries of <= 32 CTM tokens), the
// tokens' rows of X are read from a row-major copy of X in DRAM and tilized by the compute; gate_up / act / down
// run on the compact tiles; the compute scatter-adds the rows back with 0/1 placement tiles (L1 accumulation).
// As ink_moec_reader.cpp: 36 cores (core c: gate / up column tile c, down output tiles 2c, 2c + 1), bank-contiguous
// expert weights streamed through a ring, batches of NB entries, the act tiles shared through CB 13 with one
// barrier per batch.
// Pairs: from the top-k ids / weights tiles ([T, 8] of [T, 32] tiles); a zero weight (padding rows) is skipped.
// CB 0 STG (row-major staging: 2 blocks of 32 rows x 1152 columns), 1 W (weight ring), 2 D (IT x CTM act tiles),
// 6 IDX scratch, 12 CTL (compute: word 0 = U, 1 + 2j = n_j, 2 + 2j = row-tile mask of entry j), 13 SHARE (base
// only), 14 WC (CTM compact routing weight tiles), 17 LISTS (writer: word 0 = U, word 1 + j = off | n << 16, tokens
// u16 at byte 1024), 19 TMP (pairs before sorting, counts, entry experts).
// Semaphores: 0 ARRIVE (core 0), 1 GO.
// CT: 0 B, 1 NBUF, 2 MAXF, 3 MT, 4 id bytes (2 / 4), 5 D blocks, 6 staging blocks, accessors XR (row-major bf16 [T, H]), IDS, WK.
// RT: 0 xr, 1 ids, 2 wk, 3 gu base, 4 dn base, 5 c, 6 page bytes, 7 NB, 8 ncores, 9.. NoC (x, y) of each core, dbg
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t B = get_compile_time_arg_val(0), NBUF = get_compile_time_arg_val(1), MAXF = get_compile_time_arg_val(2);
constexpr uint32_t MT = get_compile_time_arg_val(3);
constexpr uint32_t IDB = get_compile_time_arg_val(4);
constexpr uint32_t DBLK = get_compile_time_arg_val(5);
constexpr uint32_t SBLK = get_compile_time_arg_val(6);  // staging blocks
constexpr uint32_t NS = 4;  // batches in SHARE (b - 2 .. b + 1 live: the pipeline)
constexpr uint32_t CTM = MT < 2 ? MT : 2;
constexpr uint32_t E = 128, KT = 72, IT = 36, T = 32 * MT, K = 8, PMAX = T * K;
constexpr uint32_t HALFB = 2304, ROWB = 4608;  // half a row / a row of X (bf16)
constexpr uint32_t NC = 36;  // cores
constexpr uint32_t TR_X = 10, TR_D = 11;  // transaction ids of the X-row and act-tile reads (ring: 1 .. NBUF)

FORCE_INLINE uint32_t elem_off(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

#define TSC() (*reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L))
#define PROF(i, ...)                \
    {                               \
        const uint32_t _t = TSC();  \
        __VA_ARGS__;                \
        prof[i] += TSC() - _t;      \
    }
uint32_t g_dbg = 0;
uint32_t prof[16];  // (profiling) wall-clock cycles per wait kind  // 1 no weight reads, 4 no act-tile gather, 8 no X reads

struct Ring {
    CircularBuffer& cw;
    uint32_t pb, base, wr = 0, inflight = 0, issued = 0;
    Ring(CircularBuffer& c, uint32_t pb_) : cw(c), pb(pb_), base(c.get_write_ptr()) {}
    void push_oldest() {
        PROF(1, noc_async_read_barrier_with_trid(1 + (issued - inflight) % NBUF));
        cw.push_back(B);
        wr = (wr + 1) % NBUF;
        --inflight;
    }
    void issue(uint32_t bank, uint32_t addr) {
        if (inflight == MAXF) push_oldest();
        PROF(2, cw.reserve_back((inflight + 1) * B));
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

FORCE_INLINE void zero_l1(uint32_t addr, uint32_t bytes) {
    const uint64_t z = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t o = 0; o < bytes; o += MEM_ZEROS_SIZE) noc_async_read(z, addr + o, MEM_ZEROS_SIZE);
    noc_async_read_barrier();
}

void kernel_main() {
    constexpr auto aX = TensorAccessorArgs<7>();
    constexpr auto aI = TensorAccessorArgs<aX.next_compile_time_args_offset()>();
    constexpr auto aW = TensorAccessorArgs<aI.next_compile_time_args_offset()>();
    const auto sX = TensorAccessor(aX, get_arg_val<uint32_t>(0));
    const auto sI = TensorAccessor(aI, get_arg_val<uint32_t>(1));
    const auto sW = TensorAccessor(aW, get_arg_val<uint32_t>(2));
    const uint32_t gu = get_arg_val<uint32_t>(3), dn = get_arg_val<uint32_t>(4), c = get_arg_val<uint32_t>(5);
    const uint32_t pb = get_arg_val<uint32_t>(6), NB = get_arg_val<uint32_t>(7), ncores = get_arg_val<uint32_t>(8);
    g_dbg = get_arg_val<uint32_t>(9 + 2 * ncores);
    const uint32_t t_start = TSC();
    for (uint32_t i = 0; i < 16; ++i) prof[i] = 0;
    // core groups: group g of NG takes the entries j with j % NG == g (its own barrier, X shares, partial output)
    const uint32_t NG = get_arg_val<uint32_t>(10 + 2 * ncores), grp = get_arg_val<uint32_t>(11 + 2 * ncores);
    constexpr uint32_t TB = 2048;
    CircularBuffer cstg(0), cw(1), cd(2), cxs(5), cidx(6), cctl(12), csh(13), cwc(14), cls(17), ctmp(19);
    // this core's share of X (rows t = c + 36 i) in L1: the other cores read the tokens' rows from there
    cxs.reserve_back(1);
    const uint32_t xs = cxs.get_write_ptr();
    for (uint32_t i = 0; c + NC * i < T; ++i) noc_async_read(sX.get_noc_addr(c + NC * i), xs + i * ROWB, ROWB);
    const uint32_t share = csh.get_read_ptr();
    // ---- (token, expert, weight) pairs in token order, then sorted by expert (stable: tokens ascending)
    ctmp.reserve_back(1);
    volatile tt_l1_ptr uint8_t* pe = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(ctmp.get_write_ptr());
    volatile tt_l1_ptr uint16_t* pt = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ctmp.get_write_ptr() + PMAX);
    volatile tt_l1_ptr uint16_t* pw = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ctmp.get_write_ptr() + 3 * PMAX);
    // small tables in L1 (the RISC-V stack is small)
    volatile tt_l1_ptr uint16_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ctmp.get_write_ptr() + 5 * PMAX);
    volatile tt_l1_ptr uint16_t* pos = cnt + E;
    volatile tt_l1_ptr uint8_t* ee = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(cnt + 2 * E);  // <= 512 entries
    for (uint32_t e = 0; e < E; ++e) cnt[e] = 0;
    uint32_t np = 0;
    cidx.reserve_back(2);
    const uint32_t ib = cidx.get_write_ptr(), wb = ib + 4096;
    for (uint32_t m = 0; m < MT; ++m) {
        noc_async_read(sI.get_noc_addr(m), ib, IDB * 1024);
        noc_async_read(sW.get_noc_addr(m), wb, TB);
        noc_async_read_barrier();
        volatile tt_l1_ptr uint16_t* w16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(wb);
        volatile tt_l1_ptr uint16_t* i16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ib);
        volatile tt_l1_ptr uint32_t* i32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ib);
        for (uint32_t r = 0; r < 32; ++r)
            for (uint32_t k = 0; k < K; ++k) {
                const uint32_t off = elem_off(r, k);
                const uint16_t w = w16[off];
                if (!(w & 0x7FFF)) continue;
                const uint32_t e = (IDB == 4 ? i32[off] : i16[off]) & (E - 1);
                pe[np] = e;
                pt[np] = 32 * m + r;
                pw[np] = w;
                ++cnt[e];
                ++np;
            }
    }
    cls.reserve_back(1);
    const uint32_t lb = cls.get_write_ptr();
    volatile tt_l1_ptr uint32_t* table = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lb);
    volatile tt_l1_ptr uint16_t* tok = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(lb + 1024);
    volatile tt_l1_ptr uint16_t* ws = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(lb + 1024 + 2 * PMAX);
    {
        uint32_t s = 0;
        for (uint32_t e = 0; e < E; ++e) {
            pos[e] = s;
            s += cnt[e];
        }
    }
    for (uint32_t i = 0; i < np; ++i) {
        const uint32_t p = pos[pe[i]]++;
        tok[p] = pt[i];
        ws[p] = pw[i];
    }
    // ---- entries: (expert, up to 32 CTM of its tokens)
    cctl.reserve_back(1);
    volatile tt_l1_ptr uint32_t* ctl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cctl.get_write_ptr());
    uint32_t U = 0, ja = 0;
    {
        uint32_t o = 0;
        for (uint32_t e = 0; e < E; ++e) {
            uint32_t left = cnt[e];
            while (left) {
                const uint32_t n = left < 32 * CTM ? left : 32 * CTM;
                if (ja++ % NG == grp) {
                    uint32_t mask = 0;
                    for (uint32_t r = 0; r < n; ++r) mask |= 1u << (tok[o + r] >> 5);
                    table[1 + U] = o | (n << 16);
                    ctl[1 + 2 * U] = n;
                    ctl[2 + 2 * U] = mask;
                    ee[U++] = e;
                }
                o += n;
                left -= n;
            }
        }
    }
    ctl[0] = U;
    table[0] = U;
    cctl.push_back(1);
    cls.push_back(1);
    const uint32_t first = (c / 4) % 2;
    const uint32_t slot = c / 4;
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    volatile tt_l1_ptr uint32_t* arr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    auto release = [&](uint32_t round) {  // coordinator: round r done once ncores * r arrivals
        noc_semaphore_wait_min(arr, ncores * round);
        *go = round;
        for (uint32_t i = 1; i < ncores; ++i)
            noc_semaphore_set_remote(get_semaphore(1), get_noc_addr(get_arg_val<uint32_t>(9 + 2 * i),
                                                                    get_arg_val<uint32_t>(10 + 2 * i), get_semaphore(1)));
        noc_async_write_barrier();
    };
    // round 1: every core's X share is in L1 (the writers' arrivals for batch b make round b + 2)
    noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(9), get_arg_val<uint32_t>(10), get_semaphore(0)), 1);
    noc_async_atomic_barrier();
    if (c == 0) release(1);
    PROF(5, noc_semaphore_wait_min(go, 1));
    prof[8] = TSC() - t_start;  // pairs, lists, X share, round 1
    Ring ring(cw, pb);
    const uint32_t nbatch = (U + NB - 1) / NB;
    // software pipeline as the compute: gate_up weights of batch b + 1, then the down of batch b
    auto gate_up_batch = [&](uint32_t b) {
        const uint32_t j0 = b * NB, j1 = (j0 + NB < U) ? j0 + NB : U;
        for (uint32_t j = j0; j < j1; ++j) {
            const uint32_t e = ee[j];
            const uint32_t o = table[1 + j] & 0xFFFF, n = table[1 + j] >> 16, ct = (n + 31) / 32;
            // (the tokens' rows of X: the writer)
            cwc.reserve_back(CTM);  // column 0 row r = routing weight of token L[r] (0 beyond n)
            volatile tt_l1_ptr uint16_t* wc = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cwc.get_write_ptr());
            for (uint32_t r = 0; r < 32 * ct; ++r) wc[(r >> 5) * 1024 + elem_off(r & 31, 0)] = r < n ? ws[o + r] : 0;
            cwc.push_back(CTM);
            for (uint32_t hh = 0; hh < 2; ++hh) {
                const uint32_t h = hh ^ first;
                const uint32_t base = gu + (e * 9 + slot) * 72 * pb;
                for (uint32_t g = 0; g < 72 / B; ++g) ring.issue(c % 4 + 4 * h, base + g * B * pb);
            }
        }
    };
    auto down_batch = [&](uint32_t b) {
        const uint32_t j0 = b * NB, j1 = (j0 + NB < U) ? j0 + NB : U;
        ring.flush();  // the compute needs its gate_up weights while this core waits at the barrier
        const uint32_t base0 = dn + (ee[j0] * 9 + slot) * 36 * pb;
        for (uint32_t g = 0; g < MAXF - 1 && g < 36 / B; ++g) ring.issue(c % 4 + 4 * first, base0 + g * B * pb);
        if (c == 0) release(b + 2);  // coordinator: release batch b once every core's writer arrived
        PROF(5, noc_semaphore_wait_min(go, b + 2));
        for (uint32_t j = j0; j < j1; ++j) {
            const uint32_t e = ee[j];
            const uint32_t ct = ((table[1 + j] >> 16) + 31) / 32;
            const uint32_t sa = share + ((b % NS) * NB + (j - j0)) * CTM * TB;  // this entry's act tiles, every core
            // the compute frees a D block only after its down: push the weights first
            if (!cd.pages_reservable_at_back(IT * CTM)) ring.flush();
            PROF(6, cd.reserve_back(IT * CTM));  // compact tile (m, kt) at m * IT + kt; m < ct read
            const uint32_t dl = cd.get_write_ptr();
            noc_async_read_set_trid(TR_D);
            for (uint32_t kt = 0; kt < IT && !(g_dbg & 4); ++kt)
                for (uint32_t m = 0; m < ct; ++m)
                    noc_async_read(get_noc_addr(get_arg_val<uint32_t>(9 + 2 * kt), get_arg_val<uint32_t>(10 + 2 * kt),
                                                sa + m * TB), dl + (m * IT + kt) * TB, TB);
            PROF(7, noc_async_read_barrier_with_trid(TR_D));
            cd.push_back(IT * CTM);
            for (uint32_t hh = 0; hh < 2; ++hh) {
                const uint32_t h = hh ^ first;
                const uint32_t base = dn + (e * 9 + slot) * 36 * pb;
                for (uint32_t g = 0; g < 36 / B; ++g) {
                    if (j == j0 && hh == 0 && g < MAXF - 1) continue;  // prefetched
                    ring.issue(c % 4 + 4 * h, base + g * B * pb);
                }
            }
        }
    };
    if (nbatch) gate_up_batch(0);
    for (uint32_t b = 0; b < nbatch; ++b) {
        if (b + 1 < nbatch) gate_up_batch(b + 1);
        down_batch(b);
    }
    ring.flush();
    if (c == 0) release(nbatch + 2);  // final round: every core has read every share slot and X row
    PROF(5, noc_semaphore_wait_min(go, nbatch + 2));
    cxs.push_back(1);
    const uint32_t pa = get_arg_val<uint32_t>(12 + 2 * ncores);
    if (pa) {  // (profiling) [0] total, [9] U, [10] batches -> row c * 3 of PROF
        prof[0] = TSC() - t_start;
        prof[9] = U;
        prof[10] = nbatch;
        volatile tt_l1_ptr uint32_t* pl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cidx.get_write_ptr());
        for (uint32_t i = 0; i < 16; ++i) pl[i] = prof[i];
        noc_async_write(cidx.get_write_ptr(), get_noc_addr_from_bank_id<true>(0, pa + (grp * 36 + c) * 3 * 64), 64);
        noc_async_write_barrier();
    }
    ctmp.push_back(1);
    cidx.push_back(2);
}

// Inkling message gate of one token in ONE program (NC = 24 cores) — reader.
//   phase A (core c): K tiles of [h | m] (h for c < NC/2, m after) and the matching D tiles -> partial z (row 0 of
//     RT tiles) and partial sum of squares, packed by the compute into CB 4 (PA, float32), shared in place;
//   barrier; every core pulls row 0 of every core's PA into dense float32 tiles (row r = core r) — 5 small reads per
//   core — and sums them with one selector matmul (compute);
//   phase B: its CPT output column tiles: the F / G tiles and those tiles of h and m.
// A final barrier round keeps the PA CBs alive until every core has read them.
// CB 0 V (KT), 1 D (KT * RT), 2 SC (reduce scaler), 5 FG (2 RT CPT), 6 H (CPT), 7 M (CPT), 8 DENSE (RT + 1, f32),
//    9 SEL (2: row 0 = ones over the h cores / the m cores), 4 PA (only its base address is used here).
// Semaphores 0 ARRIVE (core 0), 1 GO.  CT: accessors H, M, D, FG, SC.
// RT: 0 h, 1 m, 2 d, 3 fg, 4 sc, 5 c, 6 first V tile, 7 is m, 8 first D row tile, 9 KT, 10 RT, 11 j0, 12 CPT, 13 NCOL,
//     14 HT, 15 NC, 16.. NoC (x, y) of every core
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

FORCE_INLINE void release(uint32_t ncores, uint32_t value, uint32_t xy0) {
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), ncores * value);
    *go = value;
    for (uint32_t i = 1; i < ncores; ++i)
        noc_semaphore_set_remote(get_semaphore(1), get_noc_addr(get_arg_val<uint32_t>(xy0 + 2 * i),
                                                                get_arg_val<uint32_t>(xy0 + 2 * i + 1), get_semaphore(1)));
    noc_async_write_barrier();
}

void kernel_main() {
    constexpr auto aH = TensorAccessorArgs<0>();
    constexpr auto aM = TensorAccessorArgs<aH.next_compile_time_args_offset()>();
    constexpr auto aD = TensorAccessorArgs<aM.next_compile_time_args_offset()>();
    constexpr auto aF = TensorAccessorArgs<aD.next_compile_time_args_offset()>();
    constexpr auto aC = TensorAccessorArgs<aF.next_compile_time_args_offset()>();
    const auto sH = TensorAccessor(aH, get_arg_val<uint32_t>(0));
    const auto sM = TensorAccessor(aM, get_arg_val<uint32_t>(1));
    const auto sD = TensorAccessor(aD, get_arg_val<uint32_t>(2));
    const auto sF = TensorAccessor(aF, get_arg_val<uint32_t>(3));
    const auto sC = TensorAccessor(aC, get_arg_val<uint32_t>(4));
    const uint32_t c = get_arg_val<uint32_t>(5), t0 = get_arg_val<uint32_t>(6), is_m = get_arg_val<uint32_t>(7);
    const uint32_t k0 = get_arg_val<uint32_t>(8), KT = get_arg_val<uint32_t>(9), RT = get_arg_val<uint32_t>(10);
    const uint32_t j0 = get_arg_val<uint32_t>(11), CPT = get_arg_val<uint32_t>(12), NCOL = get_arg_val<uint32_t>(13);
    const uint32_t HT = get_arg_val<uint32_t>(14), NC = get_arg_val<uint32_t>(15);
    constexpr uint32_t XY = 16;
    constexpr uint32_t TB = 2048;
    Noc noc;
    CircularBuffer cv(0), cd(1), cs(2), cpa(4), cf(5), ch(6), cm(7), cdn(8), csel(9);
    const uint32_t pa = cpa.get_read_ptr();
    // phase A inputs
    cs.reserve_back(1);
    noc.async_read(sC, cs, TB, {.page_id = 0}, {.offset_bytes = 0});
    cv.reserve_back(KT);
    for (uint32_t j = 0; j < KT; ++j) {
        if (is_m) noc.async_read(sM, cv, TB, {.page_id = t0 + j}, {.offset_bytes = j * TB});
        else noc.async_read(sH, cv, TB, {.page_id = t0 + j}, {.offset_bytes = j * TB});
    }
    cd.reserve_back(KT * RT);
    for (uint32_t j = 0; j < KT; ++j)
        for (uint32_t r = 0; r < RT; ++r)
            noc.async_read(sD, cd, TB, {.page_id = (k0 + j) * RT + r}, {.offset_bytes = (j * RT + r) * TB});
    noc.async_read_barrier();
    cs.push_back(1);
    cv.push_back(KT);
    cd.push_back(KT * RT);
    // phase B inputs (independent of phase A)
    cf.reserve_back(2 * RT * CPT);
    for (uint32_t g = 0; g < 2; ++g)
        for (uint32_t k = 0; k < RT; ++k)
            for (uint32_t j = 0; j < CPT; ++j)
                noc.async_read(sF, cf, TB, {.page_id = k * NCOL + g * HT + j0 + j},
                               {.offset_bytes = ((g * RT + k) * CPT + j) * TB});
    ch.reserve_back(CPT);
    cm.reserve_back(CPT);
    for (uint32_t j = 0; j < CPT; ++j) {
        noc.async_read(sH, ch, TB, {.page_id = j0 + j}, {.offset_bytes = j * TB});
        noc.async_read(sM, cm, TB, {.page_id = j0 + j}, {.offset_bytes = j * TB});
    }
    // dense tiles zeroed (rows of absent cores must be finite) and the selectors, while phase A runs
    cdn.reserve_back(RT + 1);
    const uint32_t dn = cdn.get_write_ptr();
    {
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dn);
        for (uint32_t i = 0; i < (RT + 1) * 1024; ++i) z[i] = 0;
    }
    csel.reserve_back(2);
    {
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(csel.get_write_ptr());
        for (uint32_t i = 0; i < 2 * 512; ++i) z[i] = 0;
        volatile tt_l1_ptr uint16_t* s = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(csel.get_write_ptr());
        for (uint32_t r = 0; r < NC; ++r) {  // row 0, column r (faces 0 / 1)
            const uint32_t idx = (r >> 4) * 256 + (r & 15);
            s[(r < NC / 2 ? 0 : 1024) + idx] = 0x3F80;  // 1.0
        }
    }
    csel.push_back(2);
    noc.async_read_barrier();
    cf.push_back(2 * RT * CPT);
    ch.push_back(CPT);
    cm.push_back(CPT);
    // barrier: every core's PA is ready
    if (c == 0) release(NC, 1, XY);
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1)), 1);
    // gather row 0 of every core's partials: float32 tiles, face rows of 64 bytes (faces of 1024 bytes)
    for (uint32_t r = 0; r < NC; ++r) {
        const uint32_t x = get_arg_val<uint32_t>(XY + 2 * r), y = get_arg_val<uint32_t>(XY + 2 * r + 1);
        const uint32_t drow = (r >> 4) * 2048 + (r & 15) * 64;  // row r: face 0 / 2, (face 1 / 3 at + 1024)
        for (uint32_t k = 0; k < RT; ++k) {
            for (uint32_t f = 0; f < 2; ++f)
                noc_async_read(get_noc_addr(x, y, pa + k * 4096 + f * 1024), dn + k * 4096 + drow + f * 1024, 64);
        }
        noc_async_read(get_noc_addr(x, y, pa + RT * 4096), dn + RT * 4096 + drow, 16);  // ss at [0][0]
    }
    noc_async_read_barrier();
    cdn.push_back(RT + 1);
    // final round (released once every writer has written its outputs)
    if (c == 0) release(NC, 2, XY);
}
